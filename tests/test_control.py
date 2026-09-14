"""Tests for control sequence construction (DESIGNDOC.md §2.6, L4).

DinucShuffle: verify preserved k-mer frequencies per sequence, preserved N
positions, and (statistically) destroyed higher-order structure.
GCMatched: verify the sampled pool's GC histogram matches the primary's
within tolerance.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from pystreme.control import DinucShuffle, Explicit, GCMatched, _numba_shuffle_available, resolve_control
from pystreme.sequence_store import SequenceStore


def _kmer_counts(codes: list[int], k: int) -> Counter:
    return Counter(tuple(codes[i : i + k]) for i in range(len(codes) - k + 1))


def _random_seqs(n, length, seed, gc=0.5):
    rng = np.random.default_rng(seed)
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return ["".join(rng.choice(list("ACGT"), length, p=p)) for _ in range(n)]


ENGINES = ["python"] + (["numba"] if _numba_shuffle_available() else [])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kmer", [1, 2, 3])
def test_dinuc_shuffle_preserves_kmer_frequency_per_sequence(kmer, engine):
    seqs = _random_seqs(20, 50, seed=0)
    store = SequenceStore.from_sequences(seqs)
    shuf = DinucShuffle(kmer=kmer, seed=1, engine=engine).generate(store)

    assert shuf.codes.shape == store.codes.shape
    for i in range(store.n_seqs):
        orig = store.codes[i].tolist()
        new = shuf.codes[i].tolist()
        assert _kmer_counts(orig, kmer) == _kmer_counts(new, kmer)
        # base composition (order-0) is always preserved regardless of kmer
        assert sorted(orig) == sorted(new)


@pytest.mark.parametrize("engine", ENGINES)
def test_dinuc_shuffle_preserves_first_and_last_kmer_minus_one(engine):
    # the Euler-path construction fixes the run's start/end (k-1)-mer
    seqs = _random_seqs(10, 40, seed=2)
    store = SequenceStore.from_sequences(seqs)
    shuf = DinucShuffle(kmer=3, seed=3, engine=engine).generate(store)
    for i in range(store.n_seqs):
        orig = store.codes[i].tolist()
        new = shuf.codes[i].tolist()
        assert orig[:2] == new[:2]
        assert orig[-2:] == new[-2:]


@pytest.mark.parametrize("engine", ENGINES)
def test_dinuc_shuffle_leaves_n_positions_fixed(engine):
    seqs = ["ACGTACGTNNNNACGTACGT", "GGGGCCCCNNNGGGGCCCCA", "NNACGTACGTACGTACGTNN"]
    store = SequenceStore.from_sequences(seqs)
    shuf = DinucShuffle(kmer=2, seed=4, engine=engine).generate(store)
    for i in range(store.n_seqs):
        orig_n = [j for j, c in enumerate(store.codes[i].tolist()) if c == 4]
        new_n = [j for j, c in enumerate(shuf.codes[i].tolist()) if c == 4]
        assert orig_n == new_n


@pytest.mark.parametrize("engine", ENGINES)
def test_dinuc_shuffle_destroys_higher_order_structure(engine):
    # Higher-order structure here = a specific 8-mer implanted several times
    # per sequence, well above what dinucleotide (order-1) statistics alone
    # would predict. Dinucleotide-frequency-preserving shuffling has no
    # notion of "8-mer identity", so across a corpus of many sequences most
    # implanted copies should not survive verbatim, even though every
    # sequence's own dinucleotide counts stay exact.
    motif = "AGCTTAGC"
    rng = np.random.default_rng(9)

    def implanted(bg_len, n_copies):
        bases = list(rng.choice(list("ACGT"), bg_len))
        for _ in range(n_copies):
            pos = rng.integers(0, bg_len - len(motif))
            bases[pos : pos + len(motif)] = list(motif)
        return "".join(bases)

    seqs = [implanted(80, 3) for _ in range(40)]
    store = SequenceStore.from_sequences(seqs)
    shuf = DinucShuffle(kmer=2, seed=10, engine=engine).generate(store)

    for i in range(store.n_seqs):
        assert _kmer_counts(store.codes[i].tolist(), 2) == _kmer_counts(shuf.codes[i].tolist(), 2)

    orig_count = sum(seq.count(motif) for seq in seqs)
    shuf_seqs = ["".join("ACGTN"[c] for c in row.tolist()) for row in shuf.codes]
    shuf_count = sum(seq.count(motif) for seq in shuf_seqs)
    assert shuf_count < orig_count


def test_dinuc_shuffle_too_short_run_returned_unchanged():
    store = SequenceStore.from_sequences(["AC", "GT"])  # length 2 < kmer(2)+1
    shuf = DinucShuffle(kmer=2, seed=0).generate(store)
    assert shuf.codes[0].tolist() == store.codes[0].tolist()
    assert shuf.codes[1].tolist() == store.codes[1].tolist()


def test_dinuc_shuffle_n_per_seq_repeats():
    seqs = _random_seqs(5, 30, seed=6)
    store = SequenceStore.from_sequences(seqs)
    shuf = DinucShuffle(kmer=2, seed=7).generate(store, n_per_seq=3)
    assert shuf.n_seqs == 15


def test_gc_matched_histogram_matches_primary_within_tolerance():
    rng = np.random.default_rng(0)
    pool_seqs = []
    for gc in np.linspace(0.05, 0.95, 40):
        for _ in range(30):
            p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
            pool_seqs.append("".join(rng.choice(list("ACGT"), 60, p=p)))
    pool = SequenceStore.from_sequences(pool_seqs)

    primary_seqs = []
    for _ in range(80):
        gc = rng.uniform(0.3, 0.7)
        p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        primary_seqs.append("".join(rng.choice(list("ACGT"), 60, p=p)))
    primary = SequenceStore.from_sequences(primary_seqs)

    def gc_frac(store):
        return ((store.codes == 1) | (store.codes == 2)).float().mean(dim=1).numpy()

    matched = GCMatched(pool=pool, n_bins=20).generate(primary, seed=1)
    assert matched.n_seqs == primary.n_seqs
    pg = np.sort(gc_frac(primary))
    mg = np.sort(gc_frac(matched))
    assert np.abs(pg - mg).max() < 0.1  # within one bin width (1/20 = 0.05) plus slack


def test_gc_matched_warns_that_an_absent_bin_is_not_matched_at_all():
    """A bin the pool has *nothing* in is not "exhausted": no GC match is
    possible, the substitutes come from the whole pool, and the warning has
    to say so (and report the mismatch it actually produced) rather than
    describe sampling with replacement inside an empty bin."""
    pool = SequenceStore.from_sequences(_random_seqs(20, 30, seed=0, gc=0.8))
    primary = SequenceStore.from_sequences(_random_seqs(10, 30, seed=1, gc=0.2))
    with pytest.warns(UserWarning, match="no sequence at all") as record:
        matched = GCMatched(pool=pool, n_bins=20).generate(primary, seed=0)
    assert matched.n_seqs == primary.n_seqs
    assert not any("exhausted" in str(w.message) for w in record)
    # the controls really are unmatched, and the warning quantifies it
    def gc_frac(store):
        return ((store.codes == 1) | (store.codes == 2)).float().mean(dim=1).numpy()

    assert abs(gc_frac(matched).mean() - gc_frac(primary).mean()) > 0.3
    assert "gap" in str(record[0].message)


def test_gc_matched_warns_about_exhaustion_when_the_bin_does_exist():
    """The other half of the pair: the bin exists but is too small, so the
    GC match still holds and only the with-replacement caveat applies."""
    half_gc = "ACGT" * 7 + "AC"  # exactly 15/30 GC, so every row lands in one known bin
    pool = SequenceStore.from_sequences([half_gc] * 3)
    primary = SequenceStore.from_sequences([half_gc] * 12)
    with pytest.warns(UserWarning, match="exhausted") as record:
        matched = GCMatched(pool=pool, n_bins=5).generate(primary, seed=0)
    assert matched.n_seqs == primary.n_seqs
    assert not any("no sequence at all" in str(w.message) for w in record)


def test_gc_matched_match_repeats_not_implemented():
    pool = SequenceStore.from_sequences(_random_seqs(5, 20, seed=0))
    with pytest.raises(NotImplementedError):
        GCMatched(pool=pool, match_repeats=True)


def test_explicit_with_sequence_store():
    store = SequenceStore.from_sequences(_random_seqs(5, 20, seed=0))
    other = SequenceStore.from_sequences(_random_seqs(3, 20, seed=1))
    result = Explicit(other).generate(store)
    assert result is other


def test_explicit_with_string_list():
    store = SequenceStore.from_sequences(_random_seqs(5, 20, seed=0))
    seqs = ["ACGT" * 5, "TTTT" * 5]
    result = Explicit(seqs).generate(store)
    assert result.n_seqs == 2
    np.testing.assert_array_equal(result.codes[0].numpy(), store.__class__.from_sequences([seqs[0]]).codes[0].numpy())


def test_explicit_with_index_array_subsets_given_store():
    store = SequenceStore.from_sequences(_random_seqs(10, 20, seed=0))
    store.fit_background(order=1)
    idx = np.array([2, 4, 6])
    result = Explicit(idx).generate(store)
    assert result.n_seqs == 3
    torch.testing.assert_close(result.codes, store.codes[idx])
    # background carried over from the subset, not dropped
    assert result.bg_cumsum is not None
    torch.testing.assert_close(result.bg_cumsum, store.bg_cumsum[idx])


def test_resolve_control_dispatch():
    store = SequenceStore.from_sequences(_random_seqs(6, 20, seed=0))
    assert isinstance(resolve_control("dinuc_shuffle", store), SequenceStore)
    with pytest.raises(ValueError, match="needs a background pool"):
        resolve_control("gc_matched", store)
    with pytest.raises(ValueError, match="unknown control string"):
        resolve_control("not_a_real_mode", store)
    # a bare list of strings is wrapped in Explicit automatically
    result = resolve_control(["ACGTACGTACGTACGTACGT"], store)
    assert result.n_seqs == 1


def test_resolve_control_seed_makes_shuffle_reproducible():
    # L7 (determinism): the same seed gives the same control set, a
    # different seed a different one, and an instance's own seed wins.
    store = SequenceStore.from_sequences(_random_seqs(10, 40, seed=1))
    a = resolve_control("dinuc_shuffle", store, seed=3)
    b = resolve_control("dinuc_shuffle", store, seed=3)
    c = resolve_control("dinuc_shuffle", store, seed=4)
    assert torch.equal(a.codes, b.codes)
    assert not torch.equal(a.codes, c.codes)
    own = resolve_control(DinucShuffle(seed=3), store, seed=99)
    assert torch.equal(own.codes, a.codes)
    unseeded = resolve_control(DinucShuffle(), store, seed=3)
    assert torch.equal(unseeded.codes, a.codes)


def test_resolve_control_accepts_raw_index_array():
    # a numpy array (not a plain list) must not hit `control == "dinuc_shuffle"`
    # -- that elementwise comparison raises "ambiguous truth value" for an
    # array with more than one element.
    store = SequenceStore.from_sequences(_random_seqs(6, 20, seed=0))
    result = resolve_control(np.array([0, 1, 2]), store)
    assert result.n_seqs == 3


@pytest.mark.skipif(not _numba_shuffle_available(), reason="numba not installed")
def test_numba_shuffle_is_seeded_and_actually_shuffles():
    seqs = _random_seqs(30, 60, seed=5)
    store = SequenceStore.from_sequences(seqs)
    a = DinucShuffle(seed=8, engine="numba").generate(store)
    b = DinucShuffle(seed=8, engine="numba").generate(store)
    c = DinucShuffle(seed=9, engine="numba").generate(store)
    assert torch.equal(a.codes, b.codes)
    assert not torch.equal(a.codes, c.codes)
    assert not torch.equal(a.codes, store.codes)
    # both engines are the same algorithm: identical per-sequence dinucleotide counts
    py = DinucShuffle(seed=8, engine="python").generate(store)
    for i in range(store.n_seqs):
        assert _kmer_counts(a.codes[i].tolist(), 2) == _kmer_counts(py.codes[i].tolist(), 2)


def test_dinuc_shuffle_rejects_unknown_engine():
    with pytest.raises(ValueError, match="engine"):
        DinucShuffle(engine="fortran")


# --- regressions for the 2026-09-08 review --------------------------------


def test_resolve_control_seeding_keeps_the_requested_engine():
    """Adding a seed to an unseeded DinucShuffle must not reconfigure it:
    the replacement used to drop `engine`, so a deliberately requested
    'python' implementation silently became numba once it went through
    discovery."""
    control = DinucShuffle(kmer=3, engine="python")
    store = SequenceStore.from_sequences(_random_seqs(4, 20, seed=0))
    resolved = resolve_control(control, store, seed=7)
    assert isinstance(resolved, SequenceStore)
    # the instance resolve_control built is not observable, so check the
    # dispatch the same way it is built
    seeded = DinucShuffle(kmer=control.kmer, seed=7, engine=control.engine)
    assert seeded.engine == "python"
    assert seeded.kmer == 3
    torch.testing.assert_close(resolved.codes, seeded.generate(store).codes)


@pytest.mark.skipif(_numba_shuffle_available(), reason="numba is installed, so the guard cannot fire")
def test_dinuc_shuffle_numba_engine_requires_numba():
    store = SequenceStore.from_sequences(_random_seqs(2, 20, seed=0))
    with pytest.raises(ImportError, match="needs numba"):
        DinucShuffle(engine="numba").generate(store)


def test_explicit_index_array_resolves_against_the_universe_when_given():
    """`MotifDiscovery.enrichment(idx=..., control=[i, ...])` documents `i`
    as a row of the *whole* store; without a universe the index would hit
    the primary subset instead -- out of range at best, a different
    sequence at worst."""
    store = SequenceStore.from_sequences(_random_seqs(6, 20, seed=0))
    primary = store.subset(np.array([0, 1]))
    result = Explicit(np.array([4, 5])).generate(primary, universe=store)
    assert result.n_seqs == 2
    torch.testing.assert_close(result.codes, store.codes[np.array([4, 5])])
    # and resolve_control threads it through
    result = resolve_control(np.array([4, 5]), primary, universe=store)
    torch.testing.assert_close(result.codes, store.codes[np.array([4, 5])])
