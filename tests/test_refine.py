"""Tests for the PWM refinement loop (DESIGNDOC.md §2.1, §3 Refiner). Phase 3.

`_aligned_site_counts` (the PWM re-estimation step) gets an independent
naive reference per the project's test-before-optimize convention; the
end-to-end `refine()` recovery test is a lightweight proxy for L5 (real
Tomtom-based synthetic recovery is deferred alongside the L1/L6 MEME-suite
validation already tracked as open work).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pystreme.refine import (
    RefinedMotif,
    _aligned_site_counts,
    column_information,
    extend_flanks,
    refine,
    seed_to_pwm,
    trim_flanks,
)
from pystreme.sequence_store import SequenceStore

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _revcomp_str(word: str) -> str:
    return word.translate(_COMPLEMENT)[::-1]


def test_seed_to_pwm_shape_and_sharpness():
    pwm = seed_to_pwm("ACGT", sharpness=0.85)
    assert pwm.shape == (4, 4)
    np.testing.assert_allclose(pwm.sum(axis=0), 1.0)
    for i, c in enumerate("ACGT"):
        assert pwm["ACGT".index(c), i] == pytest.approx(0.85)


def test_column_information_bits():
    pwm = np.array([[1.0, 0.25, 0.5], [0.0, 0.25, 0.5], [0.0, 0.25, 0.0], [0.0, 0.25, 0.0]])
    np.testing.assert_allclose(column_information(pwm), [2.0, 0.0, 1.0], atol=1e-6)


def test_trim_flanks_drops_flat_columns_only_at_the_ends():
    flat = np.full(4, 0.25)
    sharp = np.array([0.91, 0.03, 0.03, 0.03])
    pwm = np.stack([flat, flat, sharp, flat, sharp, flat], axis=1)  # flat inside stays
    trimmed, n_left, n_right = trim_flanks(pwm, min_ic=0.3)
    assert (n_left, n_right) == (2, 1)
    np.testing.assert_array_equal(trimmed, pwm[:, 2:5])
    # never below min_width
    trimmed, n_left, n_right = trim_flanks(pwm, min_ic=0.3, min_width=5)
    assert trimmed.shape[1] == 5 and (n_left, n_right) == (1, 0)
    # nothing to trim -> untouched
    sharp_pwm = np.stack([sharp, sharp], axis=1)
    assert trim_flanks(sharp_pwm, min_ic=0.3)[1:] == (0, 0)


def test_extend_flanks_recovers_a_missing_edge_column():
    # implant a 10-mer, hand the extender its 8-column core (missing one
    # real column on each side); it must grow to exactly the 10-mer, and
    # stop there (the bases beyond are random), and respect max_width.
    motif = "GGGGCGGGGG"
    rng = np.random.default_rng(6)
    seqs = []
    for _ in range(120):
        bases = list(rng.choice(list("ACGT"), 60))
        pos = rng.integers(2, 60 - len(motif) - 2)  # keep room to widen on both sides
        bases[pos : pos + len(motif)] = list(motif)
        seqs.append("".join(bases))
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=0)

    core = seed_to_pwm(motif[1:-1], sharpness=0.95)
    grown = extend_flanks(core, threshold=0.0, store=store, min_ic=0.3, max_width=15, kernel="gather")
    assert grown.shape[1] == 10
    assert "".join("ACGT"[b] for b in grown.argmax(axis=0)) == motif

    capped = extend_flanks(core, threshold=0.0, store=store, min_ic=0.3, max_width=9, kernel="gather")
    assert capped.shape[1] == 9

    untouched = extend_flanks(core, threshold=0.0, store=store, min_ic=0.3, max_width=8, kernel="gather")
    assert untouched is core


def _naive_aligned_site_counts(codes, positions, strands, passing, width, pseudocount=0.1):
    counts = np.full((4, width), pseudocount)
    for n in range(len(positions)):
        if not passing[n]:
            continue
        pos = int(positions[n])
        window = codes[n, pos : pos + width].tolist()
        if strands[n] < 0:
            window = [3 - c for c in reversed(window)]
        for j, c in enumerate(window):
            counts[c, j] += 1
    return counts / counts.sum(axis=0, keepdims=True)


def test_aligned_site_counts_matches_naive_reference():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 30)) for _ in range(8)]
    store = SequenceStore.from_sequences(seqs)
    width = 5
    positions = torch.tensor([rng.integers(0, 30 - width) for _ in range(8)])
    strands = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1], dtype=torch.int8)
    passing = torch.tensor([True, True, False, True, True, False, True, True])

    actual = _aligned_site_counts(store, positions, strands, passing, width)
    expected = _naive_aligned_site_counts(store.codes.numpy(), positions.numpy(), strands.numpy(), passing.numpy(), width)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_aligned_site_counts_returns_none_when_nothing_passes():
    store = SequenceStore.from_sequences(["ACGTACGTACGT"])
    positions = torch.tensor([0])
    strands = torch.tensor([1], dtype=torch.int8)
    passing = torch.tensor([False])
    assert _aligned_site_counts(store, positions, strands, passing, 4) is None


def test_refine_requires_seed_words():
    store = SequenceStore.from_sequences(["ACGT" * 5])
    store.fit_background(order=1)
    with pytest.raises(ValueError, match="at least one seed word"):
        refine([], store, store)


def test_refine_requires_uniform_width():
    store = SequenceStore.from_sequences(["ACGT" * 5])
    store.fit_background(order=1)
    with pytest.raises(ValueError, match="same width"):
        refine(["ACGT", "ACGTA"], store, store)


def test_refine_recovers_implanted_motif():
    motif = "GGGGCGGGGG"
    rng = np.random.default_rng(5)

    def implanted(bg_len, p):
        bases = list(rng.choice(list("ACGT"), bg_len))
        if rng.random() < p:
            pos = rng.integers(0, bg_len - len(motif))
            bases[pos : pos + len(motif)] = list(motif)
        return "".join(bases)

    primary = SequenceStore.from_sequences([implanted(60, 0.7) for _ in range(150)])
    control = SequenceStore.from_sequences([implanted(60, 0.0) for _ in range(150)])
    primary.fit_background(order=1, source=control)
    control.fit_background(order=1)

    # Start from a noisy seed (3 wrong bases, not the exact answer) rather
    # than a from-scratch guess -- refinement has to do real work, but a
    # single seed mostly unrelated to the true motif (e.g. >half wrong) is
    # not what top_seeds would ever hand refine() in practice (its
    # candidates are themselves already enrichment-ranked, not random), and
    # asking a single seed with no diversity to escape that large a basin
    # in a fixed iteration budget is a different (and much harder) question
    # than "does refinement improve a plausible seed" -- discover()'s own
    # test covers the realistic multi-seed, top_seeds-driven path.
    noisy_seed = motif[:7] + "AAA"
    result = refine([noisy_seed], primary, control, n_iter=15, kernel="gather")

    assert isinstance(result, RefinedMotif)
    recovered = result.pwm.argmax(axis=0)
    recovered_word = "".join("ACGT"[b] for b in recovered)
    assert recovered_word in (motif, _revcomp_str(motif))
    assert result.n_sites > 80  # ~70% of 150 implants, minus some noise tolerance
