"""Null calibration: do the reported p-values mean what they say?

Under a null with no motif, a valid p-value satisfies P(p <= alpha) <= alpha.
Each check here counts how often that fails across many seeded null runs
and asks whether the count is *significantly* above alpha (binomial tail
below `_REJECT`), so the assertion is about calibration, not luck. Seeds
are fixed, so the outcome is deterministic.

The `fit`-based checks take ~1 min and are marked `slow`, which the
default `pytest` run skips; run them with `pytest -m slow`.
"""

from __future__ import annotations

import contextlib
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.stats import binom

from pystreme.control import DinucShuffle
from pystreme.discovery import MotifDiscovery
from pystreme.sequence_store import SequenceStore
from pystreme.statistics import central_enrichment

_ALPHAS = (0.05, 0.1, 0.25)
_REJECT = 1e-3  # binomial tail below which "more often than alpha" is not chance
_N_RUNS = 100


def _excess_tail(pvalues, alpha: float) -> float:
    """P(X >= observed) for X ~ Binomial(n, alpha): small means p <= alpha
    happened more often than a calibrated test allows."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    k = int((pvalues <= alpha).sum())
    return float(binom.sf(k - 1, pvalues.size, alpha))


def _assert_calibrated(pvalues, label: str):
    for alpha in _ALPHAS:
        tail = _excess_tail(pvalues, alpha)
        frac = float(np.mean(np.asarray(pvalues) <= alpha))
        assert tail > _REJECT, f"{label}: P(p <= {alpha}) = {frac:.2f} over {len(pvalues)} null runs (binomial tail {tail:.1e})"


# --- central enrichment, directly (fast) -----------------------------------


@pytest.mark.parametrize("n_windows", [71, 190])
def test_central_enrichment_is_calibrated_for_uniform_sites(n_windows):
    rng = np.random.default_rng(n_windows)
    centres = np.arange(n_windows) - (n_windows - 1) / 2
    pvalues = []
    for _ in range(2000):
        n_sites = int(rng.integers(20, 400))
        result = central_enrichment(rng.choice(centres, n_sites), n_windows=n_windows)
        pvalues.append(np.exp(result.log_pvalue))
    _assert_calibrated(pvalues, f"central_enrichment, {n_windows} windows")


# --- fit on data with no motif (slow) ---------------------------------------


def _null_fit(run: int, duplicated: bool, row_split: bool = False):
    """One `fit` round on iid sequences against their own dinucleotide
    shuffle: there is nothing to find, so the first motif is noise."""
    rng = np.random.default_rng(1000 + run)
    n_unique = 60 if duplicated else 120
    seqs = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(n_unique)]
    if duplicated:
        seqs = seqs + seqs
    store = SequenceStore.from_sequences(seqs)
    disc = MotifDiscovery(
        store, widths=[6, 8], control=DinucShuffle(seed=run, engine="python"), holdout_frac=0.3, seed=run
    )
    # the old behaviour: split row by row, ignoring that rows are copies.
    # Patched on the class, since fit works on a view of the store.
    row_wise = mock.patch.object(SequenceStore, "holdout_groups", lambda self: np.arange(self.n_seqs))
    with warnings.catch_warnings(), (row_wise if row_split else contextlib.nullcontext()):
        warnings.simplefilter("ignore")
        (motif,) = disc.fit(
            n_motifs=1, patience=1, keep_insignificant=True, n_per_width=2, n_iter=5, bg_order=1, kernel="conv"
        )
    return motif


@pytest.fixture(scope="module")
def null_runs():
    cache = {}

    def get(duplicated: bool, row_split: bool = False):
        key = (duplicated, row_split)
        if key not in cache:
            cache[key] = [_null_fit(r, duplicated, row_split) for r in range(_N_RUNS)]
        return cache[key]

    return get


@pytest.mark.slow
def test_holdout_pvalue_is_calibrated_under_the_null(null_runs):
    motifs = null_runs(duplicated=False)
    _assert_calibrated([np.exp(m.holdout_logp) for m in motifs], "holdout_logp, iid null")


@pytest.mark.slow
def test_training_pvalue_is_not_a_significance_measure(null_runs):
    """The training p-value is the best of every seed, iteration, width and
    threshold tried on those same sequences. Under the null it is
    'significant' nearly every time -- which is why `fit` decides on the
    hold-out, and why `holdout_frac=0` (training p only) is not safe for
    deciding whether a motif is real."""
    motifs = null_runs(duplicated=False)
    assert np.mean([np.exp(m.log_pvalue) <= 0.05 for m in motifs]) > 0.5


@pytest.mark.slow
def test_central_logp_is_calibrated_under_the_null(null_runs):
    motifs = [m for m in null_runs(duplicated=False) if m.central is not None]
    assert len(motifs) > 0.9 * _N_RUNS
    _assert_calibrated([np.exp(m.central_logp) for m in motifs], "central_logp, iid null")


@pytest.mark.slow
def test_holdout_stays_calibrated_when_every_sequence_appears_twice(null_runs):
    """Copies of one sequence are kept on one side of the split, so the
    hold-out still tests the motif on sequences the search never saw."""
    motifs = null_runs(duplicated=True)
    _assert_calibrated([np.exp(m.holdout_logp) for m in motifs], "holdout_logp, duplicated null")


@pytest.mark.slow
def test_row_wise_split_of_duplicates_is_anticonservative(null_runs):
    """The check above has power: splitting copies row by row (what `fit`
    did before hold-out groups) puts a training sequence's twin in the
    hold-out, and the hold-out p-value is then far too small."""
    pvalues = [np.exp(m.holdout_logp) for m in null_runs(duplicated=True, row_split=True)]
    assert _excess_tail(pvalues, 0.1) < _REJECT
    assert _excess_tail(pvalues, 0.25) < _REJECT


@pytest.mark.slow
def test_decoy_central_logp_is_calibrated_under_the_null(null_runs):
    motifs = [m for m in null_runs(duplicated=False) if m.decoy_central_logp is not None]
    assert len(motifs) > 0.9 * _N_RUNS
    _assert_calibrated([np.exp(m.decoy_central_logp) for m in motifs], "decoy_central_logp, iid null")
    _assert_calibrated([np.exp(m.decoy_central.log_pvalue) for m in motifs], "decoy comparison alone, iid null")
