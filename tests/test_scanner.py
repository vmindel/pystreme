"""Scanner tests: correctness against a naive brute-force reference (per the
project's test-before-optimize convention), kernel agreement (L1's "run both
scan kernels against each other"), reverse complement, and N-masking.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pystreme.scanner import scan
from pystreme.sequence_store import SequenceStore


def _make_store(seqs: list[str], device: str = "cpu") -> SequenceStore:
    return SequenceStore.from_sequences(seqs, device=device)


def _random_pwm(width, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((4, width)) + 0.05
    return raw / raw.sum(axis=0, keepdims=True)


def _random_seq(length, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT")) for _ in range(length))


def _naive_scan(codes, mask, bg_ll, log_pwm, revcomp):
    """Independent, unvectorized reference: only the scoring/argmax/revcomp
    logic is under test here -- background correctness is already covered by
    test_sequence_store.py, so bg_ll is taken as given rather than
    re-derived.
    """
    n_seqs, length = codes.shape
    w = log_pwm.shape[1]
    rc_log_pwm = log_pwm[::-1, ::-1]
    best_score = np.full(n_seqs, -np.inf)
    best_pos = np.zeros(n_seqs, dtype=int)
    best_strand = np.ones(n_seqs, dtype=int)
    for n in range(n_seqs):
        for i in range(length - w + 1):
            if not mask[n, i : i + w].all():
                continue
            bgw = bg_ll[n, i : i + w].sum()
            fwd = sum(log_pwm[codes[n, i + j], j] for j in range(w)) - bgw
            if revcomp:
                rc = sum(rc_log_pwm[codes[n, i + j], j] for j in range(w)) - bgw
                score = max(fwd, rc)
                strand = 1 if fwd >= rc else -1
            else:
                score = fwd
                strand = 1
            if score > best_score[n]:
                best_score[n] = score
                best_pos[n] = i
                best_strand[n] = strand
    return best_score, best_pos, best_strand


@pytest.fixture
def random_store():
    seqs = [_random_seq(30, seed=s) for s in range(12)]
    store = _make_store(seqs)
    store.fit_background(order=2)
    return store


@pytest.mark.parametrize("kernel", ["conv", "gather"])
@pytest.mark.parametrize("revcomp", [True, False])
def test_scan_matches_naive_reference(random_store, kernel, revcomp):
    pwm = _random_pwm(5, seed=42)
    result = scan([pwm], random_store, revcomp=revcomp, kernel=kernel)

    bg_ll = torch.diff(random_store.bg_cumsum, dim=1).numpy()
    exp_score, exp_pos, exp_strand = _naive_scan(
        random_store.codes.numpy(), random_store.mask.numpy(), bg_ll, np.log(pwm), revcomp
    )

    np.testing.assert_allclose(result.scores[0].numpy(), exp_score, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(result.positions[0].numpy(), exp_pos)
    if revcomp:
        np.testing.assert_array_equal(result.strands[0].numpy(), exp_strand)


def test_kernel_agreement(random_store):
    pwms = [_random_pwm(5, seed=1), _random_pwm(8, seed=2)]
    conv_result = scan(pwms, random_store, kernel="conv")
    gather_result = scan(pwms, random_store, kernel="gather")

    torch.testing.assert_close(conv_result.scores, gather_result.scores, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(conv_result.positions, gather_result.positions)
    torch.testing.assert_close(conv_result.strands, gather_result.strands)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["conv", "gather"])
def test_cuda_kernels_match_cpu_reference(kernel):
    """L1's fp16-vs-fp32 half: the CUDA conv kernel stores one-hot in fp16
    (§3.1) and the gather kernel runs fp32 lookups on device; both must
    agree with the fp32 CPU gather scan on a realistic 200 bp store to
    score tolerance, with identical argmax on (nearly) every sequence."""
    seqs = [_random_seq(200, seed=s) for s in range(300)]
    cpu_store = _make_store(seqs)
    cpu_store.fit_background(order=2)
    gpu_store = _make_store(seqs, device="cuda")
    gpu_store.fit_background(order=2)
    pwms = [_random_pwm(8, seed=3), _random_pwm(12, seed=4), _random_pwm(15, seed=5)]

    ref = scan(pwms, cpu_store, kernel="gather")
    got = scan(pwms, gpu_store, kernel=kernel)

    assert got.scores.device.type == "cuda"
    torch.testing.assert_close(got.scores.cpu(), ref.scores, rtol=1e-3, atol=2e-2)
    same_pos = (got.positions.cpu() == ref.positions).float().mean().item()
    assert same_pos > 0.99, same_pos
    same_strand = (got.strands.cpu() == ref.strands).float().mean().item()
    assert same_strand > 0.99, same_strand


def test_reverse_complement_detection():
    # motif strongly prefers A at every position; embedding its exact
    # reverse complement (TTTT) should only be found via the RC channel.
    pwm = np.array(
        [
            [0.94, 0.94, 0.94, 0.94],
            [0.02, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02],
        ]
    )
    seqs = [
        "GGCGG" + "AAAA" + "CCGCC",  # forward site
        "GGCGG" + "TTTT" + "CCGCC",  # reverse-complement site, no AAAA anywhere
    ]
    store = _make_store(seqs)
    store.fit_background(order=0)

    result = scan([pwm], store, revcomp=True, kernel="gather")
    assert result.strands[0, 0].item() == 1
    assert result.positions[0, 0].item() == 5
    assert result.strands[0, 1].item() == -1
    assert result.positions[0, 1].item() == 5
    assert result.scores[0, 0].item() == pytest.approx(result.scores[0, 1].item(), abs=1e-4)


def test_scan_masks_windows_spanning_n():
    pwm = np.array(
        [
            [0.94, 0.94, 0.94, 0.94],
            [0.02, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02],
            [0.02, 0.02, 0.02, 0.02],
        ]
    )
    # the only AAAA in the sequence has an N poked into it -- must not be reported
    seqs = ["GGCGG" + "AANA" + "CCGCC"]
    store = _make_store(seqs)
    store.fit_background(order=0)

    result = scan([pwm], store, revcomp=False, kernel="gather")
    assert result.positions[0, 0].item() != 5
    assert torch.isfinite(result.scores[0, 0])  # some other (weaker) window still wins


def test_scan_requires_background():
    store = _make_store(["ACGTACGTACGT"])
    with pytest.raises(RuntimeError, match="fit_background"):
        scan([_random_pwm(4, seed=0)], store)


def test_scan_idx_subset(random_store):
    pwm = _random_pwm(5, seed=7)
    idx = torch.tensor([1, 3, 5])
    subset_result = scan([pwm], random_store, idx=idx, kernel="gather")
    full_result = scan([pwm], random_store, kernel="gather")

    torch.testing.assert_close(subset_result.scores[0], full_result.scores[0, idx])
    torch.testing.assert_close(subset_result.positions[0], full_result.positions[0, idx])


@pytest.mark.parametrize("kernel", ["conv", "gather"])
def test_scan_multiple_widths_matches_individual_calls(random_store, kernel):
    pwm5 = _random_pwm(5, seed=3)
    pwm8 = _random_pwm(8, seed=4)

    combined = scan([pwm5, pwm8], random_store, kernel=kernel)
    only5 = scan([pwm5], random_store, kernel=kernel)
    only8 = scan([pwm8], random_store, kernel=kernel)

    torch.testing.assert_close(combined.scores[0], only5.scores[0])
    torch.testing.assert_close(combined.scores[1], only8.scores[0])
    torch.testing.assert_close(combined.positions[0], only5.positions[0])
    torch.testing.assert_close(combined.positions[1], only8.positions[0])


# --- tie-breaking: by rule, not by rounding ---------------------------------


def test_pick_best_breaks_near_ties_leftmost_whatever_the_rounding_noise():
    from pystreme.scanner import _pick_best

    base = torch.tensor([[[1.0, 5.0, 2.0, 5.0, 0.0]]])
    for noise in (1e-6, -1e-6):  # the later copy rounds a hair higher, or lower
        fwd = base.clone()
        fwd[0, 0, 3] += noise
        score, pos, strand = _pick_best(fwd, None)
        assert pos.item() == 1 and strand.item() == 1
        assert score.item() == fwd[0, 0, 1].item()  # the chosen window's own score
    fwd = base.clone()
    fwd[0, 0, 3] += 1e-2  # a real difference still wins
    assert _pick_best(fwd, None)[1].item() == 3


def test_pick_best_prefers_forward_on_a_strand_tie_and_reverse_on_a_real_win():
    from pystreme.scanner import _pick_best

    fwd = torch.tensor([[[0.0, 4.0, 1.0]]])
    for noise in (1e-6, -1e-6):  # a palindrome: both strands score the same
        _, pos, strand = _pick_best(fwd, fwd + noise)
        assert (pos.item(), strand.item()) == (1, 1)
    rc = fwd.clone()
    rc[0, 0, 2] = 4.5
    score, pos, strand = _pick_best(fwd, rc)
    assert (pos.item(), strand.item(), score.item()) == (2, -1, 4.5)


def test_pick_best_rows_without_a_valid_window():
    from pystreme.scanner import _pick_best

    fwd = torch.full((2, 1, 4), float("-inf"))
    fwd[1, 0, 2] = 3.0
    score, pos, strand = _pick_best(fwd, fwd.clone())
    assert score[0, 0].item() == float("-inf") and pos[0, 0].item() == 0 and strand[0, 0].item() == 1
    assert (score[1, 0].item(), pos[1, 0].item()) == (3.0, 2)


@pytest.mark.parametrize("kernel", ["conv", "gather"])
def test_scan_reports_the_first_of_two_identical_sites(kernel):
    # the golden two_motifs case: a site next to a chance copy of itself
    rng = np.random.default_rng(0)
    site = "GGGGCGGGGC"
    seqs = []
    for _ in range(20):
        bases = list(rng.choice(list("AT"), 60))  # A/T flanks: no chance near-copies
        bases[20:36] = list("C" + site + "GGGGC")  # the site at 21 and again at 26, both preceded by C
        seqs.append("".join(bases))
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=1)
    pwm = np.full((4, len(site)), 0.02)
    for i, b in enumerate(site):
        pwm["ACGT".index(b), i] = 0.94
    res = scan([pwm], store, kernel=kernel)
    assert set(res.positions[0].tolist()) == {21}


# --- the per-store masked-background cache ----------------------------------


def _fresh_scan(pwm, store, kernel):
    """The same scan on a store that has never cached anything."""
    clone = store.subset(np.arange(store.n_seqs))
    return scan([pwm], clone, kernel=kernel)


@pytest.mark.parametrize("kernel", ["conv", "gather"])
def test_cached_background_follows_erase_and_refit(kernel):
    rng = np.random.default_rng(3)
    store = SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), 40)) for _ in range(30)])
    store.fit_background(order=1)
    pwm = rng.dirichlet(np.ones(4), size=6).T
    scan([pwm], store, kernel=kernel)  # populate the cache

    first = scan([pwm], store, kernel=kernel)
    # fit's _erase_motif path: erase, then scan again under the same (now
    # stale) background tensor -- the cache key's tensor identity still
    # matches, only the mask/codes version counters have moved
    cumsum = store.bg_cumsum
    store.erase(np.arange(10), first.positions[0][:10], 6)
    store.bg_cumsum = cumsum
    after = scan([pwm], store, kernel=kernel)
    assert bool(torch.isinf(after.scores).any()) or not torch.equal(after.positions, first.positions)
    torch.testing.assert_close(after.scores, _fresh_scan(pwm, store, kernel).scores)

    store.fit_background(order=2)  # so does a new background
    torch.testing.assert_close(scan([pwm], store, kernel=kernel).scores, _fresh_scan(pwm, store, kernel).scores)


def test_cached_background_on_a_view_follows_its_owner_being_erased():
    rng = np.random.default_rng(4)
    owner = SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), 40)) for _ in range(20)])
    owner.fit_background(order=1)
    view = owner.view()
    pwm = rng.dirichlet(np.ones(4), size=5).T
    before = scan([pwm], view, kernel="gather")
    owner.erase(np.arange(20), before.positions[0], 5)  # writes into the codes/mask the view shares
    after = scan([pwm], view, kernel="gather")  # the view still holds its background tensor
    assert not torch.equal(after.scores, before.scores)
    torch.testing.assert_close(after.scores, _fresh_scan(pwm, view, "gather").scores)
