"""Decoy central enrichment: is a motif central beyond its composition?

The uniform-null test (`central_enrichment`) cannot tell a centrally bound
factor from a motif whose composition matches the peak summits: base
composition changes towards the centre of most peak sets, and a matrix with
that composition finds its best windows there. The decoy test compares the
motif's sites with those of column-permuted copies of its own matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import binom, fisher_exact

from pystreme import MotifDiscovery
from pystreme.control import DinucShuffle
from pystreme.sequence_store import SequenceStore
from pystreme.statistics import central_enrichment, decoy_central_enrichment, decoy_pwms

_REJECT = 1e-3


def _centres(n_windows):
    return np.arange(n_windows) - (n_windows - 1) / 2


# --- the statistic ----------------------------------------------------------


@pytest.mark.parametrize("n_windows", [21, 40])
def test_matches_a_per_half_width_fisher_loop(n_windows):
    rng = np.random.default_rng(n_windows)
    centres = _centres(n_windows)
    weights = np.exp(-np.abs(centres) / 4)
    offsets = rng.choice(centres, 60, p=weights / weights.sum())
    decoys = rng.choice(centres, 400)
    got = decoy_central_enrichment(offsets, decoys, n_windows)

    hs = np.unique(np.abs(centres))[:-1]
    logps = []
    for h in hs:
        a = int((np.abs(offsets) <= h).sum())
        b = int((np.abs(decoys) <= h).sum())
        _, p = fisher_exact([[a, offsets.size - a], [b, decoys.size - b]], alternative="greater")
        logps.append(np.log(p) + np.log(hs.size))
    best = int(np.argmin(logps))
    assert got.half_width == hs[best]
    assert got.log_pvalue == pytest.approx(min(logps[best], 0.0), rel=1e-9, abs=1e-9)
    assert (got.n_sites, got.decoy_sites) == (60, 400)
    assert got.fold == pytest.approx((got.n_central / 60) / (got.decoy_central / 400))


def test_none_without_sites_on_either_side():
    assert decoy_central_enrichment([], [1.0, 2.0], 10) is None
    assert decoy_central_enrichment([1.0], [], 10) is None


def test_calibrated_when_composition_alone_makes_sites_central():
    """Motif and decoy sites drawn from one centre-peaked distribution --
    the situation composition creates. The decoy test holds its level; the
    uniform-null test on the very same sites rejects almost every time."""
    n_windows = 91
    centres = _centres(n_windows)
    weights = 1 + 2 * np.exp(-((centres / 15) ** 2))  # a GC core: sites three times likelier at the centre
    weights /= weights.sum()
    rng = np.random.default_rng(0)
    decoy_p, uniform_p = [], []
    for _ in range(1000):
        n = int(rng.integers(30, 300))
        sites = rng.choice(centres, n, p=weights)
        decoys = rng.choice(centres, 10 * n, p=weights)
        decoy_p.append(np.exp(decoy_central_enrichment(sites, decoys, n_windows).log_pvalue))
        uniform_p.append(np.exp(central_enrichment(sites, n_windows).log_pvalue))
    for alpha in (0.05, 0.1, 0.25):
        k = int((np.asarray(decoy_p) <= alpha).sum())
        assert binom.sf(k - 1, len(decoy_p), alpha) > _REJECT, f"decoy test P(p<={alpha}) = {k / len(decoy_p):.3f}"
    assert np.mean(np.asarray(uniform_p) <= 0.05) > 0.5


def test_decoy_pwms_permute_columns_and_never_return_the_motif():
    rng = np.random.default_rng(1)
    pwm = rng.dirichlet(np.ones(4), size=8).T
    decoys = decoy_pwms(pwm, 25, np.random.default_rng(2))
    assert len(decoys) == 25
    for d in decoys:
        assert not np.array_equal(d, pwm)
        assert sorted(map(tuple, d.T)) == sorted(map(tuple, pwm.T))  # the same columns, reordered
    again = decoy_pwms(pwm, 25, np.random.default_rng(2))
    assert all(np.array_equal(a, b) for a, b in zip(decoys, again))
    flat = np.full((4, 5), 0.25)
    assert all(np.array_equal(d, flat) for d in decoy_pwms(flat, 3, rng))  # nothing to permute: no endless redraw
    assert decoy_pwms(pwm[:, :1], 2, rng)[0].shape == (4, 1)


# --- end to end, on peaks whose centre is GC-rich ----------------------------

_L = 200


def _gc_cored_peaks(word=None, central=False, n=600, frac=0.4, seed=0):
    rng = np.random.default_rng(seed)
    gc = np.where(np.abs(np.arange(_L) - _L / 2) < 40, 0.75, 0.4)
    probs = np.stack([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2], axis=1)
    seqs = []
    for _ in range(n):
        bases = ["ACGT"[i] for i in (rng.random((_L, 1)) > probs.cumsum(axis=1)).sum(axis=1)]
        if word is not None and rng.random() < frac:
            if central:
                pos = _L // 2 - len(word) // 2 + int(rng.integers(-3, 4))
            else:
                pos = int(rng.integers(0, _L - len(word)))
            bases[pos : pos + len(word)] = list(word)
        seqs.append("".join(bases))
    return seqs


def _pwm(word, p):
    m = np.full((4, len(word)), (1 - p) / 3)
    for i, b in enumerate(word):
        m["ACGT".index(b), i] = p
    return m


def _report(seqs, pwm):
    """Score `pwm` the way `fit` reports a motif: control, background,
    enrichment-optimal threshold, then `_finalize`."""
    disc = MotifDiscovery(SequenceStore.from_sequences(seqs), control=DinucShuffle(seed=1, engine="python"), seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primary, control = disc._prepare_primary_and_control(disc.control, None, 2, 1, 0)
        best = disc._rescore(pwm, "seed", primary, control, True, "conv")
        return disc._finalize(best, primary, None, 1, True, "conv", n_decoys=10, seed=0)


def test_a_gc_rich_matrix_with_nothing_planted_is_central_only_under_the_uniform_null():
    m = _report(_gc_cored_peaks(), _pwm("GCGCCCGCGG", 0.6))
    assert m.n_sites > 100
    assert m.central_logp < np.log(1e-20)  # the trap: composition reads as central binding
    assert m.decoy_central_logp > np.log(0.05)  # its decoys go to the centre just as much


def test_a_centrally_planted_motif_is_central_beyond_its_composition():
    m = _report(_gc_cored_peaks("GGGTGGCGC", central=True), _pwm("GGGTGGCGC", 0.85))
    assert m.decoy_central_logp < np.log(1e-20)
    assert m.decoy_central.fold > 3
    assert m.decoy_central.half_width <= 5


def test_a_uniformly_placed_motif_is_not_central_even_when_its_decoys_avoid_the_centre():
    """The reason the reported value needs both nulls: this motif leans AT,
    so in GC-cored peaks its decoys are pushed out of the centre, and its
    uniformly placed sites look central next to them. Against uniform
    placement they are not, so the motif is not reported central."""
    m = _report(_gc_cored_peaks("TCACGTGA", central=False), _pwm("TCACGTGA", 0.85))
    assert m.decoy_central.log_pvalue < np.log(1e-3)  # the decoy comparison alone would call it central
    assert m.central_logp > np.log(0.05)
    assert m.decoy_central_logp == max(m.central_logp, m.decoy_central.log_pvalue)
    assert m.decoy_central_logp > np.log(0.05)


def test_n_decoys_zero_skips_the_test_and_seed_fixes_the_decoys():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(80)]
    for s in range(0, 80, 2):
        seqs[s] = seqs[s][:25] + "TGACTCAT" + seqs[s][33:]
    disc = MotifDiscovery(SequenceStore.from_sequences(seqs), widths=[8], holdout_frac=0.0, seed=3)
    kw = dict(n_per_width=2, n_iter=3, kernel="conv", bg_order=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = disc.discover(n_decoys=0, **kw)
        a = disc.discover(**kw)
        b = disc.discover(**kw)
    assert off.decoy_central is None and off.decoy_central_logp is None
    assert a.decoy_central.n_decoys == 10
    np.testing.assert_array_equal(a.decoy_central.decoy_offsets, b.decoy_central.decoy_offsets)
