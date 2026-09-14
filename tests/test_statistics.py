"""Tests for the vectorized threshold optimizer (DESIGNDOC.md §2.5, L2).

Per the "write the naive reference first" convention, `_naive_optimal_threshold`
is an explicit per-candidate `scipy.stats.fisher_exact`/`binomtest` loop -- the
oracle `optimal_threshold`'s vectorized hypergeometric call
must agree with exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest, fisher_exact

from scipy.stats import binom, hypergeom

from pystreme.statistics import (
    binom_logsf,
    central_enrichment,
    enrichment_at_threshold,
    hypergeom_logsf,
    optimal_threshold,
)


# --- fast log tails vs scipy's exact (slow) implementations ----------------


def test_hypergeom_logsf_matches_scipy_including_underflow():
    # representable region: plain log(sf); deep tail: scipy's logsf loop
    # (exact, but a Python loop) is the oracle for the series fallback
    M, N = 20000, 10000
    n = np.array([10, 200, 2000, 2000, 2000, 5000, 5000])
    k = np.array([4, 120, 1100, 1500, 1900, 3200, 4000])
    fast = hypergeom_logsf(k, M, n, N)
    exact = hypergeom.logsf(k, M, n, N)
    assert (hypergeom.sf(k, M, n, N)[3:] == 0).any()  # the fallback is actually exercised
    np.testing.assert_allclose(fast, exact, rtol=1e-9, atol=1e-9)


def test_hypergeom_logsf_handles_impossible_and_certain_tails():
    # k >= min(n, N): P(X > k) = 0 -> -inf ; k < 0: P(X > k) = 1 -> 0
    out = hypergeom_logsf(np.array([5, -1]), 20, np.array([5, 5]), 10)
    assert out[0] == -np.inf
    assert out[1] == 0.0


def test_binom_logsf_matches_scipy_including_underflow():
    n = np.array([50, 1000, 5000, 5000])
    p = np.array([0.1, 0.05, 0.02, 0.5])
    k = np.array([10, 200, 900, 4500])
    fast = binom_logsf(k, n, p)
    assert np.isinf(binom.logsf(k, n, p)[2:]).any()  # scipy's log(sf) underflows here
    # exact upper tail by direct log-space pmf sum
    exact = np.array([np.logaddexp.reduce(binom.logpmf(np.arange(ki + 1, ni + 1), ni, pi)) for ki, ni, pi in zip(k, n, p)])
    np.testing.assert_allclose(fast, exact, rtol=1e-9, atol=1e-9)


def _naive_optimal_threshold(scores_primary, scores_control):
    scores_primary = np.asarray(scores_primary, dtype=np.float64)
    scores_control = np.asarray(scores_control, dtype=np.float64)
    n_p, n_c = len(scores_primary), len(scores_control)
    candidates = np.unique(
        np.concatenate([scores_primary[np.isfinite(scores_primary)], scores_control[np.isfinite(scores_control)]])
    )
    best_log_p = None
    best = None
    for t in candidates:
        a = int((scores_primary >= t).sum())
        b = int((scores_control >= t).sum())
        table = [[a, n_p - a], [b, n_c - b]]
        _, p = fisher_exact(table, alternative="greater")
        log_p = np.log(p) if p > 0 else -np.inf
        if best_log_p is None or log_p < best_log_p:
            best_log_p = log_p
            best = (t, log_p, a, n_p - a, b, n_c - b)
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_naive_fisher_equal_n(seed):
    rng = np.random.default_rng(seed)
    primary = rng.normal(1.0, 1.0, size=60)
    control = rng.normal(0.0, 1.0, size=60)

    result = optimal_threshold(primary, control)
    t, log_p, a, a_below, b, b_below = _naive_optimal_threshold(primary, control)

    assert result.threshold == pytest.approx(t)
    assert result.log_pvalue == pytest.approx(log_p, abs=1e-9)
    assert result.table == (a, a_below, b, b_below)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_naive_fisher_unequal_n(seed):
    rng = np.random.default_rng(seed)
    primary = rng.normal(0.5, 1.0, size=40)
    control = rng.normal(0.0, 1.0, size=90)

    result = optimal_threshold(primary, control)
    t, log_p, a, a_below, b, b_below = _naive_optimal_threshold(primary, control)

    assert result.threshold == pytest.approx(t)
    if np.isfinite(log_p):
        assert result.log_pvalue == pytest.approx(log_p, abs=1e-9)
    else:
        assert not np.isfinite(result.log_pvalue)
    assert result.table == (a, a_below, b, b_below)


def test_all_primary_above_is_maximally_significant():
    primary = np.array([5.0, 6.0, 7.0, 8.0])
    control = np.array([0.0, 1.0, 2.0, 3.0])
    result = optimal_threshold(primary, control)
    assert result.primary_above == 4
    assert result.control_above == 0
    assert result.threshold <= 5.0


def test_ties_at_threshold_handled():
    # every value repeated across primary and control -- forces literal ties
    primary = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    control = np.array([1.0, 2.0, 2.0, 3.0, 3.0])
    result = optimal_threshold(primary, control)
    t, log_p, a, a_below, b, b_below = _naive_optimal_threshold(primary, control)
    assert result.threshold == pytest.approx(t)
    assert result.table == (a, a_below, b, b_below)


def test_zero_cell_degenerate_case():
    # no overlap at all between primary and control scores: complete
    # separation, the most significant possible outcome for this sample
    # size. With n_primary=n_control=3 the smallest achievable Fisher p is
    # 1/C(6,3) = 1/20 (there's only one way to put all 3 "successes" in
    # primary out of the hypergeometric's support).
    primary = np.array([10.0, 11.0, 12.0])
    control = np.array([0.0, 1.0, 2.0])
    result = optimal_threshold(primary, control)
    assert result.control_above == 0
    assert result.primary_above == 3
    assert result.log_pvalue == pytest.approx(np.log(1 / 20))


def test_minus_inf_scores_count_toward_population_never_above():
    # a sequence with no valid scan window (§2.2) scores -inf; it must count
    # toward the population size (n_primary/n_control) but never toward
    # "above threshold", even at the lowest real (finite) candidate.
    primary = np.array([-np.inf, 1.0, 2.0, 3.0])
    control = np.array([-np.inf, -np.inf, 0.5, 0.6])

    result = optimal_threshold(primary, control)
    assert result.primary_below + result.primary_above == 4
    assert result.control_below + result.control_above == 4

    t, log_p, a, a_below, b, b_below = _naive_optimal_threshold(primary, control)
    assert result.threshold == pytest.approx(t)
    assert result.table == (a, a_below, b, b_below)

    # at the lowest finite candidate, only the finite entries clear it
    lowest = optimal_threshold(np.array([-np.inf, 5.0]), np.array([-np.inf, 5.0]))
    assert lowest.primary_above == 1
    assert lowest.control_above == 1


def test_requires_nonempty_inputs():
    with pytest.raises(ValueError):
        optimal_threshold(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        optimal_threshold(np.array([1.0]), np.array([]))


def test_requires_some_finite_score():
    with pytest.raises(ValueError):
        optimal_threshold(np.array([-np.inf, -np.inf]), np.array([-np.inf]))


def _candidate_counts(primary, control):
    n = len(primary)
    candidates = np.unique(np.concatenate([primary, control]))
    p_above = n - np.searchsorted(np.sort(primary), candidates, side="left")
    c_above = n - np.searchsorted(np.sort(control), candidates, side="left")
    return candidates, p_above, c_above


def test_large_input_bounded_search_matches_exact_argmin_when_anything_is_significant():
    # above _EXACT_ALL_BELOW candidates the Fisher branch brackets every
    # cut with O(1) bounds and evaluates only the survivors exactly. The
    # contract has two halves: values are *always* valid upper bounds, and
    # the argmin is exact whenever the best cut clears the noise floor.
    # Below that floor the search stops early on purpose (ranking noise
    # exactly cost 4.8x the whole fit and changed no reported motif), so
    # all that is promised is that the answer stays non-significant too.
    from pystreme.statistics import _fisher_logsf_min_candidates, _noise_floor, hypergeom_logsf

    rng = np.random.default_rng(11)
    n = 1500
    saw_significant = saw_noise = False
    for shift in (1.0, 0.15, 0.0):
        primary = rng.normal(shift, 1, n)
        control = rng.normal(0, 1, n)
        result = optimal_threshold(primary, control)
        candidates, p_above, c_above = _candidate_counts(primary, control)
        assert candidates.size > 512
        exact = hypergeom_logsf(p_above - 1, 2 * n, p_above + c_above, n)
        bounded = _fisher_logsf_min_candidates(p_above, c_above, n, n)
        assert (bounded >= exact - 1e-9).all()  # never below the truth, signal or not

        floor = _noise_floor(candidates.size)
        if exact.min() <= floor:
            saw_significant = True
            best = int(np.argmin(exact))
            assert result.threshold == candidates[best]
            assert result.log_pvalue == pytest.approx(exact[best], abs=1e-9)
        else:
            saw_noise = True
            # no exact argmin promised, but the caller's decision is the
            # same one it would have made with the exact answer
            assert result.log_pvalue > floor
    assert saw_significant and saw_noise  # both halves of the contract exercised


def test_noise_floor_is_bonferroni_over_the_candidates_searched():
    from pystreme.statistics import _NOISE_ALPHA, _noise_floor

    assert _noise_floor(1) == pytest.approx(np.log(_NOISE_ALPHA))
    assert _noise_floor(1000) == pytest.approx(np.log(_NOISE_ALPHA / 1000))
    assert _noise_floor(0) == _noise_floor(1)  # never divides by zero


def test_bail_out_stops_early_on_noise_but_not_on_signal():
    # the whole point of the floor: a no-signal call must not resolve the
    # candidates exhaustively, while a significant one still resolves its
    # optimum exactly (there the bounds prune to a handful anyway).
    from pystreme import statistics as st

    rng = np.random.default_rng(3)
    n = 4000
    calls = {}

    def counting(*args, **kwargs):
        calls["n"] = calls.get("n", 0) + np.size(args[0])
        return real(*args, **kwargs)

    real = st.hypergeom_logsf
    for label, shift in (("noise", 0.0), ("signal", 1.0)):
        primary, control = rng.normal(shift, 1, n), rng.normal(0, 1, n)
        candidates, p_above, c_above = _candidate_counts(primary, control)
        calls.clear()
        st.hypergeom_logsf = counting
        try:
            st._fisher_logsf_min_candidates(p_above, c_above, n, n)
        finally:
            st.hypergeom_logsf = real
        calls[label] = calls.get("n", 0)
        if label == "noise":
            # first batch (256) plus at most one doubling, not ~2x8000
            assert calls[label] <= 4 * 256, f"noise resolved {calls[label]} candidates"
        else:
            # a real optimum prunes to a handful; in particular the thousands
            # of cuts at or below the mode must not be resolved at all
            assert calls[label] <= 512, f"signal resolved {calls[label]} candidates"


# --- enrichment_at_threshold (the hold-out re-test, Phase 4) -------------


def _naive_at_threshold(scores_primary, scores_control, t):
    a = int((np.asarray(scores_primary) >= t).sum())
    b = int((np.asarray(scores_control) >= t).sum())
    n_p, n_c = len(scores_primary), len(scores_control)
    _, p = fisher_exact([[a, n_p - a], [b, n_c - b]], alternative="greater")  # any set sizes
    return a, b, np.log(p)


@pytest.mark.parametrize("n_control", [40, 25])
def test_enrichment_at_threshold_matches_naive(n_control):
    rng = np.random.default_rng(7)
    primary = rng.normal(0.5, 1, 40)
    control = rng.normal(0, 1, n_control)
    primary[:3] = -np.inf  # sequences with no valid window never count as "above"
    for t in [-1.0, 0.0, 0.7, 5.0]:
        result = enrichment_at_threshold(primary, control, t)
        a, b, log_p = _naive_at_threshold(primary, control, t)
        assert result.primary_above == a
        assert result.control_above == b
        assert result.threshold == t
        assert result.log_pvalue == pytest.approx(log_p, abs=1e-9)


def test_enrichment_at_threshold_equals_optimum_at_its_own_threshold():
    rng = np.random.default_rng(8)
    primary, control = rng.normal(1, 1, 60), rng.normal(0, 1, 60)
    opt = optimal_threshold(primary, control)
    again = enrichment_at_threshold(primary, control, opt.threshold)
    assert again.table == opt.table
    assert again.log_pvalue == pytest.approx(opt.log_pvalue)


# --- central_enrichment (§2.7, CentriMo-style) ---------------------------


def _naive_central(offsets, n_windows):
    """Per candidate half-width h, count sites and window centers within h,
    one-sided binomial test, Bonferroni over the number of h tried, min."""
    centers = np.arange(n_windows) - (n_windows - 1) / 2
    hs = sorted(set(np.abs(centers)))
    if len(hs) > 1:
        hs = hs[:-1]
    best = None
    for h in hs:
        k = sum(abs(o) <= h + 1e-9 for o in offsets)
        p_null = sum(abs(c) <= h + 1e-9 for c in centers) / n_windows
        p = binomtest(k, len(offsets), p_null, alternative="greater").pvalue
        log_p = np.log(p) + np.log(len(hs))
        if best is None or log_p < best[0]:
            best = (min(log_p, 0.0), h, k)
    return best


@pytest.mark.parametrize("n_windows", [21, 20])  # odd/even parity: integer vs half-integer centers
def test_central_enrichment_matches_naive(n_windows):
    rng = np.random.default_rng(9)
    centers = np.arange(n_windows) - (n_windows - 1) / 2
    # mostly-central sites plus a uniform tail
    offsets = np.concatenate([rng.choice(centers[abs(centers) <= 3], 30), rng.choice(centers, 20)])
    result = central_enrichment(offsets, n_windows)
    log_p, h, k = _naive_central(offsets, n_windows)
    assert result.log_pvalue == pytest.approx(log_p, abs=1e-9)
    assert result.half_width == h
    assert result.n_central == k
    assert result.n_sites == 50


def test_central_enrichment_uniform_sites_not_significant():
    rng = np.random.default_rng(10)
    n_windows = 191
    offsets = rng.integers(0, n_windows, 300) - (n_windows - 1) / 2
    result = central_enrichment(offsets, n_windows)
    assert result.log_pvalue > np.log(1e-3)


def test_central_enrichment_none_without_sites():
    assert central_enrichment(np.array([]), 50) is None


def test_unequal_sizes_zero_control_hits_is_finite():
    """A cut no control sequence clears must not give p = 0: the old
    Binomial-against-control-rate branch did (rate 0), which let a
    handful of primary hits of a rare word beat every real motif when
    the two sets differed in size. Fisher's test handles it."""
    primary = np.r_[np.full(19, 5.0), np.zeros(50_000 - 19)]
    control = np.zeros(28_000)
    result = optimal_threshold(primary, control)
    assert np.isfinite(result.log_pvalue)
    assert result.log_pvalue < 0
    _, p = fisher_exact([[19, 50_000 - 19], [0, 28_000]], alternative="greater")
    assert result.log_pvalue == pytest.approx(np.log(p), rel=1e-6)
    held = enrichment_at_threshold(primary, control, 5.0)
    assert np.isfinite(held.log_pvalue)


def test_binom_logsf_accepts_scalars():
    assert binom_logsf(3, 10, 0.5) == pytest.approx(binom.logsf(3, 10, 0.5))
    assert np.ndim(binom_logsf(3, 10, 0.5)) == 0
    assert np.isfinite(binom_logsf(900, 1000, 0.5))  # underflow branch, scalar


# --- regressions for the 2026-09-08 review --------------------------------


def test_hypergeom_logsf_scalar_in_the_deep_tail():
    """Scalar inputs whose `sf` underflows must take the same series
    fallback the array path takes, not crash. `np.log` of a 0-d array is a
    `np.float64`, which cannot be assigned into by mask -- the whole reason
    `binom_logsf` reshapes to 1-D first."""
    fast = hypergeom_logsf(999, 2000, 1000, 1000)
    assert np.isfinite(fast)
    assert hypergeom.sf(999, 2000, 1000, 1000) == 0.0  # the fallback is actually exercised
    np.testing.assert_allclose(fast, hypergeom.logsf(999, 2000, 1000, 1000), rtol=1e-9)
    assert np.ndim(fast) == 0  # a scalar in, a scalar out


def test_enrichment_at_threshold_survives_a_perfectly_separating_holdout():
    """`fit`'s hold-out re-test on a motif that separates its two sets
    perfectly: the most significant result possible is exactly the one that
    used to abort the run."""
    result = enrichment_at_threshold(np.ones(1000), np.zeros(1000), 0.5)
    assert np.isfinite(result.log_pvalue)
    assert result.log_pvalue < -1000
    assert result.table == (1000, 0, 0, 1000)


def test_central_enrichment_deep_tail_stays_finite_and_picks_the_right_width():
    """Every site dead center: scipy's `binom.logsf` underflows to -inf
    here, which loses the value *and* ties every candidate width, so the
    reported half-width would be whichever came first rather than the best."""
    n_sites, n_windows = 2000, 101
    result = central_enrichment(np.zeros(n_sites), n_windows)
    n_candidates = 50  # |offsets| 0..49; the all-inclusive width is dropped
    expected = n_sites * np.log(1 / n_windows) + np.log(n_candidates)
    assert np.isfinite(result.log_pvalue)
    np.testing.assert_allclose(result.log_pvalue, expected, rtol=1e-9)
    assert result.half_width == 0.0
    assert result.n_central == n_sites


def test_bounded_fisher_search_without_signal_is_conservative_and_still_noise():
    """Where the bounds can't discriminate (no cut is significant, nearly
    every candidate survives the ceiling) the search stops early instead of
    resolving them all: an exact argmin over noise cost 4.8x the whole fit
    and changed no reported motif. What it still promises there: the value
    is never below the true optimum (an upper bound, so nothing is reported
    as more significant than it is) and it stays above the noise floor, so
    the caller discards it exactly as it would the exact answer."""
    from pystreme.statistics import _noise_floor

    rng = np.random.default_rng(5)
    primary = rng.normal(0, 1, 600)
    control = rng.normal(0, 1, 600)
    result = optimal_threshold(primary, control)
    _, log_p, *_ = _naive_optimal_threshold(primary, control)
    floor = _noise_floor(np.unique(np.concatenate([primary, control])).size)
    assert log_p > floor  # the premise: nothing here is significant
    assert result.log_pvalue >= log_p - 1e-9
    assert result.log_pvalue > floor
