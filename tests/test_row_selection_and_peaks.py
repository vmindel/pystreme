"""Regression tests for three input-handling bugs:

1. A boolean mask passed as a row selection was cast to long, i.e. turned
   into a list of 0s and 1s that selected rows 0 and 1 over and over.
2. Duplicate and overlapping peaks were kept as independent sequences and
   could land on both sides of the hold-out split.
3. narrowPeak summits (column 10) were ignored; windows were centred on the
   interval midpoint.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from pystreme.control import Explicit
from pystreme.discovery import MotifDiscovery, _holdout_split
from pystreme.scanner import scan
from pystreme.sequence_store import Interval, SequenceStore, _parse_bed, as_row_index


def _random_store(n=40, length=30, seed=0):
    rng = np.random.default_rng(seed)
    store = SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), length)) for _ in range(n)])
    store.fit_background(order=1)
    return store


@pytest.fixture
def random_genome(tmp_path):
    """Two random chromosomes, so windows at different offsets differ in
    sequence (the shared conftest genome is periodic)."""
    rng = np.random.default_rng(7)
    chroms = {"chr1": "".join(rng.choice(list("ACGT"), 2000)), "chr2": "".join(rng.choice(list("ACGT"), 1000))}
    path = tmp_path / "genome.fa"
    path.write_text("".join(f">{c}\n{s}\n" for c, s in chroms.items()))
    return path, chroms


def _write_bed(tmp_path, rows, name="peaks.bed"):
    path = tmp_path / name
    path.write_text("".join("\t".join(str(x) for x in r) + "\n" for r in rows))
    return path


# --- 1. row selection ------------------------------------------------------


def test_as_row_index_resolves_a_boolean_mask_to_its_true_rows():
    mask = np.zeros(10, dtype=bool)
    mask[[2, 5, 7]] = True
    assert as_row_index(mask, 10).tolist() == [2, 5, 7]
    assert as_row_index(torch.from_numpy(mask), 10).tolist() == [2, 5, 7]
    assert as_row_index([False, True, True], 3).tolist() == [1, 2]


def test_as_row_index_rejects_bad_selections():
    with pytest.raises(ValueError, match="boolean row mask"):
        as_row_index(np.ones(9, dtype=bool), 10)
    with pytest.raises(IndexError):
        as_row_index([0, 10], 10)
    with pytest.raises(IndexError):
        as_row_index([-1], 10)  # rejected, not wrapped to the last row
    with pytest.raises(TypeError):
        as_row_index(np.array([0.5]), 10)
    assert as_row_index([], 10).numel() == 0
    assert as_row_index(np.array([1.0, 3.0]), 10).tolist() == [1, 3]


def test_subset_with_a_mask_equals_subset_with_its_indices():
    store = _random_store()
    mask = np.zeros(store.n_seqs, dtype=bool)
    mask[10:25] = True
    by_mask, by_idx = store.subset(mask), store.subset(np.flatnonzero(mask))
    assert by_mask.n_seqs == 15
    torch.testing.assert_close(by_mask.codes, by_idx.codes)
    torch.testing.assert_close(by_mask.bg_cumsum, by_idx.bg_cumsum)
    assert [iv.name for iv in by_mask.intervals] == [f"seq_{i}" for i in range(10, 25)]


def test_scan_with_a_mask_equals_scan_with_its_indices():
    store = _random_store()
    pwm = np.random.default_rng(1).dirichlet(np.ones(4), size=6).T
    mask = np.arange(store.n_seqs) % 3 == 0
    a = scan([pwm], store, idx=mask)
    b = scan([pwm], store, idx=np.flatnonzero(mask))
    torch.testing.assert_close(a.scores, b.scores)


def test_enrichment_and_fit_accept_a_mask_as_the_primary_selection():
    store = _random_store(n=60, length=40)
    disc = MotifDiscovery(store, widths=[6], holdout_frac=0.0, seed=0)
    mask = np.zeros(store.n_seqs, dtype=bool)
    mask[::2] = True
    pwm = np.random.default_rng(2).dirichlet(np.ones(4), size=6).T
    by_mask = disc.enrichment(pwm, idx=mask, kernel="conv")
    by_idx = disc.enrichment(pwm, idx=np.flatnonzero(mask), kernel="conv")
    assert by_mask == by_idx
    assert by_mask.primary_above + by_mask.primary_below == 30

    kw = dict(n_motifs=1, keep_insignificant=True, n_per_width=2, n_iter=3, kernel="conv")
    (m_mask,) = disc.fit(primary_idx=mask, **kw)
    (m_idx,) = disc.fit(primary_idx=np.flatnonzero(mask), **kw)
    assert m_mask.sites.scores.shape == (30,)
    torch.testing.assert_close(m_mask.sites.scores, m_idx.sites.scores)


def test_explicit_control_accepts_a_mask_over_the_universe():
    store = _random_store()
    mask = np.zeros(store.n_seqs, dtype=bool)
    mask[30:] = True
    control = Explicit(mask).generate(store.subset(np.arange(5)), universe=store)
    torch.testing.assert_close(control.codes, store.codes[30:])


# --- 2. duplicate and overlapping peaks -------------------------------------


def test_from_bed_drops_identical_windows_as_duplicates(tmp_path, random_genome):
    genome, _ = random_genome
    rows = [
        ("chr1", 100, 200, "a"),
        ("chr1", 100, 200, "a_again"),  # same interval
        ("chr1", 120, 180, "same_centre"),  # different interval, same window
        ("chr1", 600, 700, "b"),
    ]
    with pytest.warns(UserWarning, match="dropped 2/4"):
        store = SequenceStore.from_bed(_write_bed(tmp_path, rows), genome, size=50)
    assert [iv.name for iv in store.intervals] == ["a", "b"]
    assert [(e.interval.name, e.reason) for e in store.excluded] == [("a_again", "duplicate"), ("same_centre", "duplicate")]


def test_from_bed_warns_about_overlapping_windows(tmp_path, random_genome):
    genome, _ = random_genome
    rows = [("chr1", 100, 200, "a"), ("chr1", 130, 230, "b"), ("chr1", 900, 1000, "c")]
    with pytest.warns(UserWarning, match="2/3 windows overlap"):
        store = SequenceStore.from_bed(_write_bed(tmp_path, rows), genome, size=100)
    assert store.n_seqs == 3
    assert store.holdout_groups().tolist() == [0, 0, 1]


def test_holdout_groups_join_identical_rows_and_overlapping_genomic_windows(tmp_path, random_genome):
    genome, _ = random_genome
    rows = [
        ("chr1", 0, 100, "a"),
        ("chr2", 0, 100, "x"),
        ("chr1", 90, 190, "b"),  # overlaps a by 10 bp
        ("chr1", 180, 280, "c"),  # overlaps b, so a-b-c is one cluster
        ("chr1", 1000, 1100, "d"),  # chr1 and chr2 both start at 0: overlap needs the same chromosome
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store = SequenceStore.from_bed(_write_bed(tmp_path, rows), genome, size=100)
    assert store.genomic
    assert store.holdout_groups().tolist() == [0, 1, 0, 0, 2]
    assert store.subset([2, 3, 4]).holdout_groups().tolist() == [0, 0, 1]  # survives subsetting

    # from_sequences intervals are all ("control", 0, L): not genomic, so
    # only identical content groups rows
    seqs = ["ACGTACGTAA", "TTTTACGTAA", "ACGTACGTAA", "GGGGCCCCAA"]
    in_memory = SequenceStore.from_sequences(seqs)
    assert not in_memory.genomic
    assert in_memory.holdout_groups().tolist() == [0, 1, 0, 2]


def test_holdout_split_with_singleton_groups_is_the_plain_permutation():
    n, frac = 200, 0.15
    plain_train, plain_hold = _holdout_split(n, frac, np.random.default_rng(3), 10)
    grouped_train, grouped_hold = _holdout_split(n, frac, np.random.default_rng(3), 10, groups=np.arange(n))
    np.testing.assert_array_equal(plain_train, grouped_train)
    np.testing.assert_array_equal(plain_hold, grouped_hold)
    perm = np.random.default_rng(3).permutation(n)
    np.testing.assert_array_equal(plain_hold, np.sort(perm[: round(frac * n)]))


def test_holdout_split_never_puts_one_group_on_both_sides():
    rng = np.random.default_rng(0)
    groups = np.sort(rng.integers(0, 80, 300))
    groups = np.unique(groups, return_inverse=True)[1]  # relabel 0.. in first-row order
    for seed in range(20):
        train, hold = _holdout_split(groups.size, 0.2, np.random.default_rng(seed), 10, groups=groups)
        assert hold is not None
        assert not set(groups[train]) & set(groups[hold])
        assert hold.size >= round(0.2 * groups.size)
        assert np.array_equal(np.sort(np.concatenate([train, hold])), np.arange(groups.size))


def test_holdout_split_is_disabled_when_groups_make_it_impossible():
    groups = np.zeros(100, dtype=np.int64)  # one group: everything shares sequence
    train, hold = _holdout_split(100, 0.1, np.random.default_rng(0), 10, groups=groups)
    assert hold is None and train.size == 100


def test_fit_keeps_copies_of_a_sequence_on_one_side_of_the_split(monkeypatch):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 40)) for _ in range(50)]
    store = SequenceStore.from_sequences(seqs + seqs)  # row i and row i+50 are copies

    seen = {}
    real_split = _holdout_split

    def spy(n, frac, rng_, min_holdout, groups=None):
        out = real_split(n, frac, rng_, min_holdout, groups=groups)
        seen.setdefault("primary", out)  # fit splits the primary first
        return out

    monkeypatch.setattr("pystreme.discovery._holdout_split", spy)
    MotifDiscovery(store, widths=[6], holdout_frac=0.2, seed=0).fit(
        n_motifs=1, keep_insignificant=True, n_per_width=2, n_iter=2, bg_order=1, kernel="conv"
    )
    train, hold = seen["primary"]
    assert hold is not None
    assert not {i % 50 for i in train} & {i % 50 for i in hold}


# --- 3. narrowPeak summits --------------------------------------------------


def _narrowpeak_row(chrom, start, end, name, summit_offset):
    return (chrom, start, end, name, 0, ".", 5.0, 10.0, 8.0, summit_offset)


def test_parse_bed_reads_the_narrowpeak_summit(tmp_path):
    rows = [
        _narrowpeak_row("chr1", 100, 600, "p1", 50),
        _narrowpeak_row("chr1", 1000, 1400, "p2", -1),  # MACS: no summit
        _narrowpeak_row("chr1", 2000, 2100, "p3", 500),  # outside the peak: not trusted
    ]
    ivs = _parse_bed(_write_bed(tmp_path, rows, "peaks.narrowPeak"))
    assert [iv.summit for iv in ivs] == [150, None, None]


def test_parse_bed_does_not_read_bed12_block_count_as_a_summit(tmp_path):
    rows = [("chr1", 100, 600, "tx", 0, "+", 100, 600, "0,0,0", 2, "50,50,", "0,450,")]
    (iv,) = _parse_bed(_write_bed(tmp_path, rows))
    assert iv.summit is None


def test_from_bed_centres_windows_on_the_summit(tmp_path, random_genome):
    genome, chroms = random_genome
    rows = [_narrowpeak_row("chr1", 100, 900, "p1", 40), _narrowpeak_row("chr2", 200, 300, "p2", -1)]
    bed = _write_bed(tmp_path, rows, "peaks.narrowPeak")
    store = SequenceStore.from_bed(bed, genome, size=60)
    assert [(iv.start, iv.end) for iv in store.intervals] == [(110, 170), (220, 280)]  # summit 140; midpoint 250
    assert store.to_strings()[0] == chroms["chr1"][110:170]
    assert store.intervals[0].summit == 140

    midpoint = SequenceStore.from_bed(bed, genome, size=60, summit="midpoint")
    assert [(iv.start, iv.end) for iv in midpoint.intervals] == [(470, 530), (220, 280)]

    with pytest.raises(ValueError, match="summit"):
        SequenceStore.from_bed(bed, genome, size=60, summit="peak")


def test_from_bed_uses_summits_on_pre_parsed_intervals(random_genome):
    genome, _ = random_genome
    ivs = [Interval("chr1", 100, 900, "p", summit=700)]
    store = SequenceStore.from_bed(ivs, genome, size=40)
    assert (store.intervals[0].start, store.intervals[0].end) == (680, 720)


def test_broad_peak_warning_skips_summit_centred_peaks(tmp_path, random_genome):
    genome, _ = random_genome
    rows = [_narrowpeak_row("chr1", 100 + 300 * i, 380 + 300 * i, f"p{i}", 140) for i in range(5)]
    bed = _write_bed(tmp_path, rows, "peaks.narrowPeak")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SequenceStore.from_bed(bed, genome, size=50)  # 280 bp peaks but summit-centred: no warning
    with pytest.warns(UserWarning, match="median input peak width"):
        SequenceStore.from_bed(bed, genome, size=50, summit="midpoint")
