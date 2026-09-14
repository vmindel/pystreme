"""Unit tests for SequenceStore: extraction, edge handling, background model.

DESIGNDOC.md L3 (k-mer/edge correctness) territory. The background-model test
follows the "write the naive reference first" convention from §2.5: a plain
nested-loop Python implementation is the oracle for the vectorized
unfold+bincount one in sequence_store.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pystreme.sequence_store import SequenceStore, encode

from conftest import CHR_A, SIZE

# genome_fasta and peaks_bed fixtures live in conftest.py (shared with
# test_discovery.py); CHR_A/SIZE imported above for the assertions below.


def test_encode():
    codes = encode("ACGTacgtN n")
    np.testing.assert_array_equal(codes, [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4])


def test_extraction_and_exclusions(genome_fasta, peaks_bed):
    with pytest.warns(UserWarning, match="dropped"):
        store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)

    kept_names = {iv.name for iv in store.intervals}
    assert kept_names == {"peak_ok", "peak_someN", "peak_chrB_ok"}

    excl_by_name = {exc.interval.name: exc.reason for exc in store.excluded}
    assert excl_by_name == {
        "peak_edge": "chrom_edge",
        "peak_toomanyN": "max_n_frac",
        "peak_unknown_chrom": "unknown_chrom",
        "peak_chrB_edge": "chrom_edge",
    }

    assert store.n_seqs == 3
    assert store.length == SIZE


def test_extracted_sequence_matches_manual_slice(genome_fasta, peaks_bed):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    ok = next(i for i, iv in enumerate(store.intervals) if iv.name == "peak_ok")
    expected = encode(CHR_A[40:60])
    np.testing.assert_array_equal(store.codes[ok].numpy(), expected)

    iv = store.intervals[ok]
    assert (iv.start, iv.end) == (40, 60)


def test_bulk_chromosome_extraction_matches_per_interval(genome_fasta, peaks_bed, monkeypatch):
    import pystreme.sequence_store as ss

    monkeypatch.setattr(ss, "_BLOCK_GAP", -(10**9))  # every window its own genome read
    with pytest.warns(UserWarning):
        per_interval = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    monkeypatch.setattr(ss, "_BLOCK_GAP", 10**9)  # one read per chromosome
    with pytest.warns(UserWarning):
        bulk = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)

    assert torch.equal(bulk.codes, per_interval.codes)
    assert torch.equal(bulk.mask, per_interval.mask)
    assert bulk.intervals == per_interval.intervals
    assert [(e.interval.name, e.reason) for e in bulk.excluded] == [
        (e.interval.name, e.reason) for e in per_interval.excluded
    ]


def test_from_bed_warns_on_broad_peaks(genome_fasta, tmp_path):
    bed = tmp_path / "broad.bed"
    bed.write_text("chrA\t20\t140\tbroad\n")  # 120 bp peak, size 20 -> median width 6x size
    with pytest.warns(UserWarning, match="median input peak width"):
        store = SequenceStore.from_bed(bed, genome_fasta, size=SIZE)
    assert store.n_seqs == 1


def test_n_masking_at_threshold(genome_fasta, peaks_bed):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    some_n = next(i for i, iv in enumerate(store.intervals) if iv.name == "peak_someN")
    row = store.codes[some_n]
    mask_row = store.mask[some_n]
    # window [141,161): only position 160 (last one) is N
    assert (row == 4).sum().item() == 1
    assert row[-1].item() == 4
    assert not mask_row[-1].item()
    assert mask_row[:-1].all()


def test_onehot_matches_codes(genome_fasta, peaks_bed):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    oh = store.onehot(dtype=torch.float32)  # (N, 4, L)
    assert oh.shape == (store.n_seqs, 4, store.length)
    for n in range(store.n_seqs):
        for i in range(store.length):
            col = oh[n, :, i]
            if store.mask[n, i]:
                assert col.sum().item() == 1
                assert col.argmax().item() == store.codes[n, i].item()
            else:
                assert col.sum().item() == 0


def _naive_background(codes: np.ndarray, mask: np.ndarray, order: int, pseudocount: float = 1.0) -> np.ndarray:
    """Plain nested-loop reference: the oracle for fit_background's vectorized path."""
    n_seqs, length = codes.shape
    counts0 = np.zeros(4)
    for n in range(n_seqs):
        for i in range(length):
            if mask[n, i]:
                counts0[codes[n, i]] += 1
    counts0 += pseudocount
    log_freq0 = np.log(counts0 / counts0.sum())

    bg_ll = np.zeros((n_seqs, length))
    if order == 0:
        for n in range(n_seqs):
            for i in range(length):
                if mask[n, i]:
                    bg_ll[n, i] = log_freq0[codes[n, i]]
        return bg_ll

    k = order
    counts = _naive_order_counts(codes, mask, k)

    def log_cond_for(j, ctx):
        c = counts[j].get(ctx, np.zeros(4)).copy() + pseudocount
        return np.log(c / c.sum())

    for n in range(n_seqs):
        for i in range(length):
            if not mask[n, i]:
                continue
            j = _naive_usable_order(mask[n], i, k)
            if j == 0:
                bg_ll[n, i] = log_freq0[codes[n, i]]
            else:
                ctx = tuple(int(c) for c in codes[n, i - j : i])
                bg_ll[n, i] = log_cond_for(j, ctx)[codes[n, i]]
    return bg_ll


def _naive_order_counts(codes: np.ndarray, mask: np.ndarray, order: int) -> dict[int, dict]:
    """One context->base-count table per order 1..order, counted only from
    windows whose context *and* base are all valid."""
    n_seqs, length = codes.shape
    tables: dict[int, dict] = {}
    for j in range(1, order + 1):
        counts: dict[tuple, np.ndarray] = {}
        for n in range(n_seqs):
            for i in range(j, length):
                if mask[n, i - j : i + 1].all():
                    ctx = tuple(int(c) for c in codes[n, i - j : i])
                    counts.setdefault(ctx, np.zeros(4))
                    counts[ctx][codes[n, i]] += 1
        tables[j] = counts
    return tables


def _naive_usable_order(row_mask: np.ndarray, i: int, order: int) -> int:
    """How much context position `i` actually has: the length of the run of
    valid bases immediately before it, capped at `order` (and at `i`)."""
    j = 0
    while j < order and i - (j + 1) >= 0 and row_mask[i - (j + 1) : i].all():
        j += 1
    return j


@pytest.mark.parametrize("order", [0, 1, 2])
def test_fit_background_matches_naive_reference(genome_fasta, peaks_bed, order):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    store.fit_background(order=order)

    expected = _naive_background(store.codes.numpy(), store.mask.numpy(), order=order)
    actual = torch.diff(store.bg_cumsum, dim=1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_background_window_sum(genome_fasta, peaks_bed):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    store.fit_background(order=2)
    bg_ll = torch.diff(store.bg_cumsum, dim=1)

    start, width = 3, 8
    expected = bg_ll[:, start : start + width].sum(dim=1)
    actual = store.background_window_sum(start, width)
    torch.testing.assert_close(actual, expected)


def test_background_requires_longer_than_order(genome_fasta, peaks_bed):
    store = SequenceStore.from_bed(peaks_bed, genome_fasta, size=SIZE)
    with pytest.raises(ValueError, match="requires sequences longer"):
        store.fit_background(order=SIZE)  # order == length, not > it


def _naive_background_from_source(
    fit_codes: np.ndarray,
    fit_mask: np.ndarray,
    apply_codes: np.ndarray,
    apply_mask: np.ndarray,
    order: int,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Like `_naive_background`, but the table is counted from one array
    pair (`fit_*`, standing in for `source`) and applied to a *different*
    array pair (`apply_*`, standing in for `self`) -- the oracle for
    `fit_background(order, source=...)` when source and self differ in
    shape (§2.3's "fit on the control set" recommendation, used by
    `MotifDiscovery.enrichment`).
    """
    n_fit, len_fit = fit_codes.shape
    counts0 = np.zeros(4)
    for n in range(n_fit):
        for i in range(len_fit):
            if fit_mask[n, i]:
                counts0[fit_codes[n, i]] += 1
    counts0 += pseudocount
    log_freq0 = np.log(counts0 / counts0.sum())

    n_apply, len_apply = apply_codes.shape
    bg_ll = np.zeros((n_apply, len_apply))
    if order == 0:
        for n in range(n_apply):
            for i in range(len_apply):
                if apply_mask[n, i]:
                    bg_ll[n, i] = log_freq0[apply_codes[n, i]]
        return bg_ll

    k = order
    counts = _naive_order_counts(fit_codes, fit_mask, k)  # tables from source

    def log_cond_for(j, ctx):
        c = counts[j].get(ctx, np.zeros(4)).copy() + pseudocount
        return np.log(c / c.sum())

    for n in range(n_apply):
        for i in range(len_apply):
            if not apply_mask[n, i]:
                continue
            j = _naive_usable_order(apply_mask[n], i, k)  # see _naive_background
            if j == 0:
                bg_ll[n, i] = log_freq0[apply_codes[n, i]]
            else:
                ctx = tuple(int(c) for c in apply_codes[n, i - j : i])
                bg_ll[n, i] = log_cond_for(j, ctx)[apply_codes[n, i]]
    return bg_ll


@pytest.mark.parametrize("order", [0, 1, 2])
def test_fit_background_with_different_source_matches_naive_reference(order):
    # source (the "control") and self (the "primary") differ in both n_seqs
    # and length -- fit_background must still apply source's table to self's
    # own codes/mask, not source's.
    rng = np.random.default_rng(0)

    def make_store(n, length):
        seqs = ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]
        return SequenceStore.from_sequences(seqs)

    primary = make_store(6, 25)
    control = make_store(15, 40)

    primary.fit_background(order=order, source=control)

    expected = _naive_background_from_source(
        control.codes.numpy(), control.mask.numpy(), primary.codes.numpy(), primary.mask.numpy(), order=order
    )
    actual = torch.diff(primary.bg_cumsum, dim=1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_fit_background_source_equals_self_by_default_unchanged():
    # regression guard: source=None must still behave exactly as before the
    # cross-store fix (both paths reduce to the same computation).
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGT"), 30)) for _ in range(8)]
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=2)
    expected = _naive_background(store.codes.numpy(), store.mask.numpy(), order=2)
    actual = torch.diff(store.bg_cumsum, dim=1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_subset_keeps_fitted_background_indexed_consistently():
    rng = np.random.default_rng(2)
    seqs = ["".join(rng.choice(list("ACGT"), 30)) for _ in range(10)]
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=2)

    idx = [1, 3, 5]
    sub = store.subset(idx)
    assert sub.n_seqs == 3
    assert sub.bg_order == 2
    torch.testing.assert_close(sub.bg_cumsum, store.bg_cumsum[idx])
    torch.testing.assert_close(sub.codes, store.codes[idx])


def test_subset_without_fitted_background_stays_unfitted():
    rng = np.random.default_rng(3)
    seqs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(5)]
    store = SequenceStore.from_sequences(seqs)
    sub = store.subset([0, 1])
    assert sub.bg_cumsum is None


# --- erase (STREME's site erasing, Phase 4) ------------------------------


def test_erase_masks_exactly_the_given_windows():
    rng = np.random.default_rng(4)
    seqs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(4)]
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=1)
    before = store.codes.clone()

    n = store.erase(rows=[0, 2, 2], starts=[3, 0, 15], width=5)
    assert n == 3
    expected_mask = torch.ones(4, 20, dtype=torch.bool)
    expected_mask[0, 3:8] = False
    expected_mask[2, 0:5] = False
    expected_mask[2, 15:20] = False
    assert torch.equal(store.mask, expected_mask)
    assert (store.codes[~expected_mask] == 4).all()
    assert torch.equal(store.codes[expected_mask], before[expected_mask])
    assert store.bg_cumsum is None  # must be refit

    # k-mer counting and scanning now treat those windows as absent
    store.fit_background(order=1)
    from pystreme.seeds import _kmer_ids

    _, valid = _kmer_ids(store.codes, store.mask, 4)
    assert not valid[0, 0:8].any()  # any 4-window touching [3,8) is invalid
    assert valid[0, 8:].all()


def test_erase_rejects_out_of_range_and_mismatched_inputs():
    store = SequenceStore.from_sequences(["ACGT" * 5])
    with pytest.raises(ValueError, match="outside"):
        store.erase([0], [18], 5)
    with pytest.raises(ValueError, match="same length"):
        store.erase([0, 0], [1], 3)
    assert store.erase([], [], 3) == 0


def test_erase_does_not_touch_the_parent_of_a_subset():
    store = SequenceStore.from_sequences(["ACGT" * 5, "TTTT" * 5])
    sub = store.subset([1])
    sub.erase([0], [0], 4)
    assert store.mask.all()


# --- regressions for the 2026-09-08 review --------------------------------


def test_background_after_an_internal_n_uses_the_marginal_not_a_t_context():
    """Codes are clamped (N -> T) to index the conditional table, which is
    harmless at the N itself (masked to 0) but not for the valid bases that
    follow it: those windows are scored for real and used to get P(x | ...T)
    for a T that isn't there. They must fall back to the order-0 marginal.
    """
    # source: T is always followed by C, so P(A|T) is pseudocount-only and
    # nowhere near the marginal P(A)
    source = SequenceStore.from_sequences(["TCTCTCTCTCTCTCTCTCTC"] * 4 + ["AAAACCCCGGGGAAAATTTT"] * 4)
    target = SequenceStore.from_sequences(["ACGTACGTACNAACGTACGT"])
    target.fit_background(order=1, source=source)

    after_n = 11  # the A at index 11; index 10 is the N
    value = float(target.background_window_sum(after_n, 1)[0])
    assert value == pytest.approx(float(target.bg_marginals[0]), abs=1e-6)
    assert not np.isclose(value, float(target.bg_table[3, 0]))  # not P(A | T)

    # the base *after* that one has a full valid context again
    assert float(target.background_window_sum(after_n + 1, 1)[0]) == pytest.approx(
        float(target.bg_table[0, 0]), abs=1e-6
    )


def test_background_after_an_erased_site_backs_off_one_order_at_a_time():
    """Erasing writes Ns into the middle of a sequence, so every later round
    scores windows whose context starts inside an erased site -- the same
    fabricated-context condition, reached without any N in the input.

    The back-off is graded: only the position whose *immediately* preceding
    base was erased loses all context. Dropping every partially-valid
    position to the marginal flattens the background exactly where erasing
    has left low-complexity sequence, and costs real motifs in late rounds.
    """
    store = SequenceStore.from_sequences(["ACGTACGTACGTACGTACGT"] * 4)
    store.erase(rows=[0], starts=[8], width=4)  # positions 8-11 become N
    store.fit_background(order=2)

    # position 12: context (10, 11) both erased -> no context at all
    assert float(store.background_window_sum(12, 1)[0]) == pytest.approx(
        float(store.bg_marginals[int(store.codes[0, 12])]), abs=1e-6
    )
    # position 13: context (11, 12) -- 11 erased, 12 valid -> order 1, not the marginal
    order1 = store.bg_tables[1]
    expected = float(order1[int(store.codes[0, 12]), int(store.codes[0, 13])])
    assert float(store.background_window_sum(13, 1)[0]) == pytest.approx(expected, abs=1e-6)
    assert not np.isclose(expected, float(store.bg_marginals[int(store.codes[0, 13])]))
    # position 14: context (12, 13) fully valid again -> order 2
    ctx = int(store.codes[0, 12]) * 4 + int(store.codes[0, 13])
    assert float(store.background_window_sum(14, 1)[0]) == pytest.approx(
        float(store.bg_table[ctx, int(store.codes[0, 14])]), abs=1e-6
    )


def test_background_first_positions_back_off_by_available_context():
    """The same rule at a sequence start: position 0 has no context, position
    1 has exactly one base, position 2 onwards has the full order."""
    store = SequenceStore.from_sequences(["ACGTACGTACGTACGTACGT"] * 4)
    store.fit_background(order=2)
    codes = store.codes[0]
    assert float(store.background_window_sum(0, 1)[0]) == pytest.approx(
        float(store.bg_marginals[int(codes[0])]), abs=1e-6
    )
    assert float(store.background_window_sum(1, 1)[0]) == pytest.approx(
        float(store.bg_tables[1][int(codes[0]), int(codes[1])]), abs=1e-6
    )
    assert float(store.background_window_sum(2, 1)[0]) == pytest.approx(
        float(store.bg_table[int(codes[0]) * 4 + int(codes[1]), int(codes[2])]), abs=1e-6
    )


def test_bg_marginals_available_at_every_order():
    store = SequenceStore.from_sequences(["ACGTACGTACGTACGTACGT"] * 3)
    for order in (0, 1, 2):
        store.fit_background(order=order)
        assert store.bg_marginals is not None
        assert store.bg_marginals.shape == (4,)
        np.testing.assert_allclose(float(store.bg_marginals.exp().sum()), 1.0, rtol=1e-6)


def test_background_model_applied_to_many_stores_matches_fitting_each():
    """`background_model` + `apply_background` on N stores must be exactly
    `fit_background(source=...)` on each, only estimated once -- that is the
    identity the round loop relies on to stop re-counting the same control
    set four times per round."""
    rng = np.random.default_rng(4)

    def store(n, length):
        return SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), length)) for _ in range(n)])

    control = store(12, 40)
    targets = [store(5, 40), store(7, 40), store(3, 40)]

    for order in (0, 1, 2):
        expected = []
        for t in targets:
            t.fit_background(order=order, source=control)
            expected.append(t.bg_cumsum.clone())

        model = control.background_model(order)
        for t, want in zip(targets, expected):
            t.bg_cumsum = None
            t.apply_background(model)
            torch.testing.assert_close(t.bg_cumsum, want)
            assert t.bg_order == order
            assert t.bg_model is model  # the same estimate, not a re-fit


def test_background_model_records_what_it_was_fitted_on():
    control = SequenceStore.from_sequences(["ACGTACGTACGTACGTACGT"] * 6)
    model = control.background_model(order=2)
    assert model.order == 2
    assert model.n_seqs == 6 and model.length == 20
    assert len(model.tables) == 3  # orders 0, 1, 2
    assert model.tables[0].shape == (4,) and model.tables[1].shape == (4, 4) and model.tables[2].shape == (16, 4)
    torch.testing.assert_close(model.marginals, model.tables[0])
