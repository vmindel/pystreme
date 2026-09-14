"""MotifDiscovery wiring: from_bed -> scan -> to_meme (Phase 1's
`disc.scan(jaspar_pwm, idx)` deliverable), enrichment (Phase 2's
SEA-equivalent `disc.enrichment(pwm, control=...)` deliverable), and
discover (Phase 3's single-motif de novo discovery deliverable), per
DESIGNDOC.md's public API sketch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pystreme.control import GCMatched
from pystreme.discovery import MotifDiscovery
from pystreme.meme_io import read_meme
from pystreme.scanner import ScanResult, scan
from pystreme.sequence_store import SequenceStore

from conftest import SIZE


def _random_pwm(width, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((4, width)) + 0.05
    return raw / raw.sum(axis=0, keepdims=True)


def test_from_bed_scan_single_pwm_matches_raw_scanner(genome_fasta, peaks_bed):
    disc = MotifDiscovery.from_bed(peaks_bed, genome_fasta, size=SIZE, bg_order=1)
    pwm = _random_pwm(4, seed=0)

    result = disc.scan(pwm)
    assert isinstance(result, ScanResult)
    assert result.scores.ndim == 1  # squeezed: no leading motif axis for a single pwm
    assert result.scores.shape[0] == disc.store.n_seqs

    raw = scan([pwm], disc.store)
    torch.testing.assert_close(result.scores, raw.scores[0])
    torch.testing.assert_close(result.positions, raw.positions[0])


def test_scan_list_of_pwms_keeps_motif_axis(genome_fasta, peaks_bed):
    disc = MotifDiscovery.from_bed(peaks_bed, genome_fasta, size=SIZE, bg_order=1)
    pwms = [_random_pwm(4, seed=1), _random_pwm(5, seed=2)]

    result = disc.scan(pwms)
    assert result.scores.shape == (2, disc.store.n_seqs)


def test_scan_idx_forwarded(genome_fasta, peaks_bed):
    disc = MotifDiscovery.from_bed(peaks_bed, genome_fasta, size=SIZE, bg_order=1)
    pwm = _random_pwm(4, seed=3)
    idx = torch.tensor([0, disc.store.n_seqs - 1])

    result = disc.scan(pwm, idx=idx)
    assert result.scores.shape[0] == 2


def test_to_meme_roundtrip(tmp_path):
    disc = MotifDiscovery.__new__(MotifDiscovery)  # to_meme doesn't touch self.store
    pwm = _random_pwm(6, seed=4)
    path = tmp_path / "out.meme"

    disc.to_meme({"MOTIF1": pwm}, path)

    parsed = read_meme(path)["MOTIF1"]
    np.testing.assert_allclose(parsed.pwm, pwm, atol=1e-5)


# --- enrichment() (Phase 2 deliverable: "SEA-equivalent enrichment testing,
# in-process") -----------------------------------------------------------


def _motif_pwm(motif: str, sharpness: float = 0.94) -> np.ndarray:
    pwm = np.full((4, len(motif)), (1 - sharpness) / 3)
    for i, c in enumerate(motif):
        pwm["ACGT".index(c), i] = sharpness
    return pwm / pwm.sum(axis=0, keepdims=True)


def _implanted_store(motif: str, n: int, bg_len: int, p_implant: float, seed: int) -> SequenceStore:
    rng = np.random.default_rng(seed)

    def one():
        bases = list(rng.choice(list("ACGT"), bg_len))
        if rng.random() < p_implant:
            pos = rng.integers(0, bg_len - len(motif))
            bases[pos : pos + len(motif)] = list(motif)
        return "".join(bases)

    return SequenceStore.from_sequences([one() for _ in range(n)])


def test_enrichment_detects_implanted_motif_with_dinuc_shuffle():
    motif = "GGGGCGGGGG"
    store = _implanted_store(motif, n=150, bg_len=60, p_implant=0.7, seed=0)
    disc = MotifDiscovery(store)

    result = disc.enrichment(_motif_pwm(motif), control="dinuc_shuffle", bg_order=2, kernel="gather")

    assert result.primary_above > result.control_above
    assert result.log_pvalue < np.log(1e-6)


def test_enrichment_no_signal_when_motif_absent():
    store = _implanted_store("GGGGCGGGGG", n=100, bg_len=60, p_implant=0.0, seed=1)
    disc = MotifDiscovery(store)

    result = disc.enrichment(_motif_pwm("GGGGCGGGGG"), control="dinuc_shuffle", bg_order=1, kernel="gather")

    # no implants at all: primary and control should look statistically alike
    assert result.log_pvalue > np.log(1e-3)


def test_enrichment_idx_restricts_primary_set():
    motif = "GGGGCGGGGG"
    store = _implanted_store(motif, n=100, bg_len=60, p_implant=1.0, seed=2)
    disc = MotifDiscovery(store)
    idx = np.arange(10)  # only 10 sequences count as "primary"

    result = disc.enrichment(_motif_pwm(motif), control="dinuc_shuffle", bg_order=1, idx=idx, kernel="gather")
    assert result.primary_above + result.primary_below == 10


def test_enrichment_accepts_gc_matched_control():
    motif = "GGGGCGGGGG"
    store = _implanted_store(motif, n=60, bg_len=60, p_implant=0.7, seed=3)
    pool = _implanted_store("AAAAAAAAAA", n=300, bg_len=60, p_implant=0.0, seed=4)  # no real motif in the pool
    disc = MotifDiscovery(store)

    result = disc.enrichment(
        _motif_pwm(motif), control=GCMatched(pool=pool, n_bins=10), bg_order=1, kernel="gather"
    )
    assert result.primary_above > result.control_above


def test_enrichment_requires_bg_order_when_none_available():
    store = SequenceStore.from_sequences(["ACGT" * 15 for _ in range(5)])
    disc = MotifDiscovery(store)  # no from_bed, so store.bg_order is None
    with pytest.raises(RuntimeError, match="no bg_order available"):
        disc.enrichment(_motif_pwm("ACGT"))


# --- discover() (Phase 3 deliverable: "single-motif de novo discovery") --


def test_discover_recovers_implanted_motif_at_known_width():
    # a lightweight proxy for L5's synthetic-recovery check (real
    # Tomtom-based L5/L6 comparisons are deferred alongside the other
    # MEME-suite validation work) -- implant a known motif at a majority of
    # sequences and check discover() finds it (on either strand). Fixed to
    # a single width deliberately: comparing raw enrichment p-values across
    # *different* widths isn't apples-to-apples (a shorter, less specific
    # word can look more "enriched" simply by matching more sequences
    # partially -- real STREME corrects for this with a width-aware
    # significance adjustment that discover() doesn't attempt yet, tracked
    # as Phase 4 work alongside the round loop's own width comparisons).
    # test_refine.py separately exercises recovery-from-a-noisy-seed at a
    # fixed width; this test is about discover()'s own wiring end to end.
    motif = "GGGGCGGGGG"
    store = _implanted_store(motif, n=150, bg_len=60, p_implant=0.7, seed=42)
    disc = MotifDiscovery(store)

    result = disc.discover(
        widths=[len(motif)], control="dinuc_shuffle", bg_order=1, n_per_width=3, n_iter=10, kernel="gather"
    )

    recovered = "".join("ACGTN"[b] for b in result.pwm.argmax(axis=0))
    assert recovered in (motif, motif.translate(str.maketrans("ACGT", "TGCA"))[::-1])
    assert result.width == len(motif)
    assert result.n_sites > 80
    assert result.log_pvalue < np.log(1e-10)
    assert result.holdout_logp is None  # discover() has no hold-out split; fit() does
    assert result.positions.shape == (result.n_sites,)
    assert result.central_logp is not None  # implants are uniformly placed, so this is just "computed"


def test_discover_raises_when_no_seeds_found_for_any_width():
    store = SequenceStore.from_sequences(["N" * 40 for _ in range(10)])  # no valid windows anywhere
    disc = MotifDiscovery(store)
    with pytest.raises(RuntimeError, match="no seed words found"):
        disc.discover(widths=[6, 7], control="dinuc_shuffle", bg_order=1, kernel="gather")


# --- fit() (Phase 4 deliverable: the round loop) --------------------------

_RC = str.maketrans("ACGT", "TGCA")
MOTIF_A = "GGGGCGGGGG"  # implanted near the center, in most sequences
MOTIF_B = "TGACTCATTT"  # implanted anywhere, in fewer sequences


def _argmax_word(pwm):
    return "".join("ACGT"[b] for b in pwm.argmax(axis=0))


def _matches(word, motif):
    return word in (motif, motif.translate(_RC)[::-1])


def _two_motif_store(n=300, bg_len=80, seed=0):
    rng = np.random.default_rng(seed)
    center = (bg_len - len(MOTIF_A)) // 2
    seqs = []
    for _ in range(n):
        bases = list(rng.choice(list("ACGT"), bg_len))
        if rng.random() < 0.6:
            pos = center + rng.integers(-3, 4)  # tightly centered, §2.7's ChEC-like regime
            bases[pos : pos + len(MOTIF_A)] = list(MOTIF_A)
        if rng.random() < 0.4:
            pos = rng.integers(0, bg_len - len(MOTIF_B))
            if pos + len(MOTIF_B) <= center - 3 or pos >= center + 3 + len(MOTIF_A):  # don't overwrite A
                bases[pos : pos + len(MOTIF_B)] = list(MOTIF_B)
        seqs.append("".join(bases))
    return SequenceStore.from_sequences(seqs)


_FIT_KW = dict(widths=[8, 10], bg_order=1, n_per_width=3, n_iter=10, kernel="gather")


def test_fit_finds_both_implanted_motifs_in_order_with_holdout_significance():
    disc = MotifDiscovery(_two_motif_store(), holdout_frac=0.2, seed=0)
    motifs = disc.fit(n_motifs=2, **_FIT_KW)

    assert len(motifs) == 2
    first, second = motifs
    assert first.round == 1 and second.round == 2
    # erasing worked: the second motif is a different one, and it's B
    assert _matches(_argmax_word(first.pwm), MOTIF_A), first
    assert _matches(_argmax_word(second.pwm), MOTIF_B), second
    for m in motifs:
        assert m.holdout is not None
        assert m.holdout_logp == m.holdout.log_pvalue
        assert m.holdout_logp <= np.log(0.05)
        assert m.holdout.primary_above + m.holdout.primary_below == 60  # 20% of 300 held out
        assert m.sites.scores.shape == (300,)  # sites reported on the full primary set, not just training
        assert m.positions.shape == (m.n_sites,)
        assert m.n_sites == int((m.sites.scores >= m.threshold).sum())
    # A sits at the center (§2.7): strongly central; B is placed uniformly
    assert first.central_logp < np.log(1e-6)
    assert first.central.half_width <= 5
    assert second.central_logp > first.central_logp
    assert repr(first).startswith("Motif(1-")


def test_discover_settles_the_width_across_a_range():
    # The training p-value saturates for every width from the 8-mer core
    # up to the 10-mer plus flat flanks (all within a fraction of a nat of
    # each other on this data), so the p-value alone would pick the width
    # by noise. Flank trimming + the information-content tie-break must
    # land on the implanted 10-mer, not its core or a padded version.
    disc = MotifDiscovery(_two_motif_store(), holdout_frac=0.0, seed=0)
    m = disc.discover(widths=[8, 9, 10, 11, 12], bg_order=1, n_per_width=3, n_iter=10, kernel="gather")
    assert m.width == 10, m
    assert _matches(_argmax_word(m.pwm), MOTIF_A), m


def test_fit_is_deterministic_for_a_seed():
    # L7: same seed, same output (control set, hold-out split, everything)
    store = _two_motif_store()
    a = MotifDiscovery(store, seed=11).fit(n_motifs=2, **_FIT_KW)
    b = MotifDiscovery(store, seed=11).fit(n_motifs=2, **_FIT_KW)
    assert len(a) == len(b) == 2
    for ma, mb in zip(a, b):
        np.testing.assert_array_equal(ma.pwm, mb.pwm)
        assert ma.holdout_logp == mb.holdout_logp
        assert ma.threshold == mb.threshold


def test_fit_without_holdout_uses_training_pvalue():
    disc = MotifDiscovery(_two_motif_store(n=120), seed=1)
    motifs = disc.fit(n_motifs=1, holdout_frac=0.0, **_FIT_KW)
    assert len(motifs) == 1
    assert motifs[0].holdout_logp is None
    assert motifs[0].log_pvalue <= np.log(0.05)


def test_fit_warns_and_skips_holdout_when_too_few_sequences():
    disc = MotifDiscovery(_two_motif_store(n=40), seed=1)
    with pytest.warns(UserWarning, match="hold-out disabled"):
        motifs = disc.fit(n_motifs=1, holdout_frac=0.1, **_FIT_KW)  # 4 held out < min_holdout
    assert motifs[0].holdout_logp is None


def test_fit_stops_on_insignificant_motifs_and_can_keep_them():
    store = _implanted_store(MOTIF_A, n=200, bg_len=60, p_implant=0.0, seed=5)  # nothing to find
    disc = MotifDiscovery(store, seed=2)
    assert disc.fit(n_motifs=1, patience=1, **_FIT_KW) == []
    kept = disc.fit(n_motifs=1, patience=2, keep_insignificant=True, **_FIT_KW)
    assert len(kept) == 2  # two rejected rounds, then patience ran out
    assert all(m.holdout_logp > np.log(0.05) for m in kept)


def test_fit_primary_idx_restricts_the_primary_set():
    store = _two_motif_store(n=200)
    disc = MotifDiscovery(store, holdout_frac=0.0, seed=3)
    motifs = disc.fit(primary_idx=np.arange(100), n_motifs=1, **_FIT_KW)
    assert motifs[0].sites.scores.shape == (100,)


def test_fit_rejects_bad_arguments():
    disc = MotifDiscovery(_two_motif_store(n=50))
    with pytest.raises(ValueError, match="n_motifs"):
        disc.fit(n_motifs=0)
    with pytest.raises(ValueError, match="pvalue_thresh"):
        disc.fit(pvalue_thresh=0)


def test_to_meme_accepts_motif_list(tmp_path):
    disc = MotifDiscovery(_two_motif_store(n=120), holdout_frac=0.0, seed=4)
    motifs = disc.fit(n_motifs=1, **_FIT_KW)
    path = tmp_path / "fit.meme"
    disc.to_meme(motifs, path)

    parsed = read_meme(path)
    assert list(parsed) == [motifs[0].name]
    m = parsed[motifs[0].name]
    np.testing.assert_allclose(m.pwm, motifs[0].pwm, atol=1e-5)
    assert m.nsites == motifs[0].n_sites
    assert m.evalue == pytest.approx(np.exp(motifs[0].log_pvalue), rel=1e-3)


# --- regressions for the 2026-09-08 review --------------------------------


def test_control_of_a_different_length_is_rejected():
    """Every test here compares per-sequence best scores, which is only fair
    when both sides have the same number of possible motif starts. Unequal
    lengths used to be accepted silently and biased enrichment towards the
    longer side."""
    store = SequenceStore.from_sequences(["ACGTACGTACGT"] * 6)
    store.fit_background(order=0)
    disc = MotifDiscovery(store, widths=[4])
    with pytest.raises(ValueError, match="equal lengths"):
        disc.enrichment(_random_pwm(4, seed=0), control=["ACGTAC"] * 6)


def test_explicit_control_indices_are_rows_of_the_whole_store():
    """`enrichment`'s docstring promises control indices into `self.store`;
    they used to be resolved against the `idx` subset instead, which raises
    for an in-store index out of the subset's range (and silently picks the
    wrong row when it happens to be in range)."""
    seqs = ["ACGTACGTACGT", "TTTTAAAATTTT", "GGGGCCCCGGGG", "CCCCGGGGCCCC"]
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=0)
    disc = MotifDiscovery(store, widths=[4])

    primary_idx = np.array([0, 1])
    result = disc.enrichment(_random_pwm(4, seed=0), idx=primary_idx, control=np.array([3]))
    assert result.primary_above + result.primary_below == 2
    assert result.control_above + result.control_below == 1

    # and the control really is row 3, not row 3 of the subset (which does
    # not exist) nor some other row
    _, control_store = disc._prepare_primary_and_control(np.array([3]), primary_idx, None, 1, None)
    torch.testing.assert_close(control_store.codes, store.codes[np.array([3])])


def test_fit_reports_sites_under_the_round_background_not_the_setup_one():
    """`fit` learns a threshold on the training store (background fitted on
    the training control, refit every round after erasing) but reports sites
    by scanning the full primary store. That store used to keep the setup
    background, fitted on the whole, unerased control -- a threshold from one
    scoring model applied to scores from another.

    With identical primary sequences every reported score is the same number,
    and it is the same number the threshold was chosen from, so either every
    sequence is a site or the two backgrounds disagree. It used to report
    zero sites for a motif present in all 200 of them.
    """
    seq = "ACACACACACACACACACAC" + "TTGACGTCAA" + "ACACACACACACACACACAC"
    store = SequenceStore.from_sequences([seq] * 200)
    store.fit_background(order=1)
    disc = MotifDiscovery(store, widths=[10], holdout_frac=0.1, seed=0)

    motifs = disc.fit(n_motifs=1, n_iter=5, keep_insignificant=True)
    assert motifs
    motif = motifs[0]
    assert len(set(motif.sites.scores.tolist())) == 1  # identical sequences, identical scores
    assert motif.n_sites == store.n_seqs


def test_forward_only_discovery_recovers_the_implanted_orientation():
    """End-to-end `revcomp=False`: the seed, the refinement and the scan now
    agree on one strand, so the implanted word is recovered instead of a
    near-random motif built from the seed's reverse complement."""
    word = "GTTTTC"
    rng = np.random.default_rng(12)
    seqs = []
    for _ in range(200):
        b = list(rng.choice(list("ACGT"), 40))
        pos = int(rng.integers(0, 34))
        b[pos : pos + len(word)] = list(word)
        seqs.append("".join(b))
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=1)
    disc = MotifDiscovery(store, widths=[6], revcomp=False, seed=0)

    motif = disc.discover(n_iter=10)
    assert motif.consensus == word
    assert motif.log_pvalue < -50


def test_discover_on_rescores_a_correction_that_keeps_the_original_width(monkeypatch):
    """Trimming one uninformative flank and growing an informative one on
    the other side lands back on the starting width with a different PWM.
    Re-scoring keyed on width alone missed that and returned the old motif,
    discarding the correction."""
    import pystreme.discovery as D

    rng = np.random.default_rng(13)
    word = "ACGTGTCA"
    seqs = []
    for _ in range(120):
        b = list(rng.choice(list("ACGT"), 40))
        pos = int(rng.integers(0, 32))
        b[pos : pos + len(word)] = list(word)
        seqs.append("".join(b))
    primary = SequenceStore.from_sequences(seqs)
    control = SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), 40)) for _ in range(120)])
    control.fit_background(order=1)
    primary.fit_background(order=1, source=control)

    corrected = {}

    def fake_trim(pwm, min_ic, min_width=1):
        return pwm[:, 1:], 1, 0  # drop one flank...

    def fake_extend(pwm, threshold, store, min_ic, max_width, revcomp=True, kernel="auto", control_store=None):
        out = np.full((4, pwm.shape[1] + 1), 0.1)  # ...and grow one back: same width, different matrix
        out[0, :] = 0.7
        corrected["pwm"] = out
        return out

    monkeypatch.setattr(D, "trim_flanks", fake_trim)
    monkeypatch.setattr(D, "extend_flanks", fake_extend)

    result = MotifDiscovery._discover_on(primary, control, [8], 2, 5, True, "auto")
    assert result is not None
    assert result.pwm.shape == corrected["pwm"].shape
    np.testing.assert_array_equal(result.pwm, corrected["pwm"])  # the correction, not the pre-trim PWM
