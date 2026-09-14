"""Vectorized enrichment statistics.

See DESIGNDOC.md §2.5. Phase 2. The vectorized optimizer is validated (L2) in
tests/test_statistics.py against a brute-force per-threshold
`scipy.stats.fisher_exact` loop -- that reference loop is written as a test
fixture, not thrown away, since it's the oracle for this file for good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, hypergeom


@dataclass
class ThresholdResult:
    """Result of `optimal_threshold`: the best cut point and its significance.

    `log_pvalue` is a natural-log p-value (not log10), computed via scipy's
    `logsf` rather than `log(sf(...))` so it stays finite deep into
    underflow territory instead of collapsing to `-inf` -- the whole point of
    working in log-p space here (§2.5's "single most likely way to end up
    slower" concern is about speed, but STREME-style refinement also needs
    p-values sharp enough to rank candidates once `sf` itself would return
    machine zero).
    """

    threshold: float
    log_pvalue: float
    primary_above: int
    primary_below: int
    control_above: int
    control_below: int

    @property
    def table(self) -> tuple[int, int, int, int]:
        """2x2 contingency table as (primary_above, primary_below, control_above, control_below)."""
        return (self.primary_above, self.primary_below, self.control_above, self.control_below)


_UNDERFLOW = 1e-300  # below this, log(sf) has lost its relative precision (or is -inf)


def _log_tail_series(log_p0: np.ndarray, log_ratio: np.ndarray) -> np.ndarray:
    """log( p0 * (1 + r1 + r1 r2 + r1 r2 r3 + ...) ) from log p0 and the
    per-term log ratios (n, T): the log of a tail sum whose terms fall off
    geometrically, in log space so nothing underflows."""
    cum = np.cumsum(log_ratio, axis=1)  # log(r1), log(r1 r2), ...
    return log_p0 + np.logaddexp.reduce(np.concatenate([np.zeros((cum.shape[0], 1)), cum], axis=1), axis=1)


def hypergeom_logsf(k, M, n, N, n_terms: int = 512) -> np.ndarray:
    """`log P(X > k)` for X ~ Hypergeometric(M, n, N), vectorized and fast.

    `scipy.stats.hypergeom.logsf` is a Python loop that logsumexps the
    whole upper tail *per element* (this was 95% of `fit`'s runtime, and
    the seed ranking hands it every candidate word at once -- millions on
    a real peak set). `hypergeom.sf` is a C ufunc but underflows to 0 for
    the strongly enriched words that matter most, so: `log(sf)` wherever
    `sf` is representable, and for the rest the tail series summed
    directly in log space -- `pmf(k+1)` (vectorized `logpmf`) times the
    geometric-like sum of successive pmf ratios, which by then (far above
    the mode, or `sf` wouldn't have underflowed) decay fast enough that
    `n_terms` terms are exact to double precision.
    """
    k, M, n, N = np.broadcast_arrays(*(np.asarray(a, dtype=np.int64) for a in (k, M, n, N)))
    shape = k.shape
    k, M, n, N = (a.reshape(-1) for a in (k, M, n, N))  # scalars too: the underflow branch assigns by mask
    p = hypergeom.sf(k, M, n, N)
    with np.errstate(divide="ignore"):
        out = np.log(p)
    under = p < _UNDERFLOW
    if under.any():
        kk, MM, nn, NN = (a[under] for a in (k, M, n, N))
        k1 = kk + 1  # sf(k) = P(X >= k+1)
        log_p0 = hypergeom.logpmf(k1, MM, nn, NN)
        j = np.arange(n_terms)[None, :]
        kj = k1[:, None] + j
        num = (nn[:, None] - kj) * (NN[:, None] - kj)  # pmf(x+1)/pmf(x) numerator at x = kj
        den = (kj + 1) * (MM[:, None] - nn[:, None] - NN[:, None] + kj + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(num > 0, np.log(num) - np.log(den), -np.inf)
        out[under] = _log_tail_series(log_p0, log_ratio)
    return out.reshape(shape)


def binom_logsf(k, n, p, n_terms: int = 512) -> np.ndarray:
    """`log P(X > k)` for X ~ Binomial(n, p): `log(binom.sf)` (a C ufunc)
    with the same log-space tail-series fallback as `hypergeom_logsf`
    where `sf` underflows (scipy's `binom.logsf` is just `log(sf)`, so it
    returns `-inf` there)."""
    k, n, p = np.broadcast_arrays(np.asarray(k, dtype=np.int64), np.asarray(n, dtype=np.int64), np.asarray(p, dtype=np.float64))
    shape = k.shape
    k, n, p = k.reshape(-1), n.reshape(-1), p.reshape(-1)  # scalars too: the underflow branch assigns by mask
    sf = binom.sf(k, n, p)
    with np.errstate(divide="ignore"):
        out = np.log(sf)
    under = sf < _UNDERFLOW
    if under.any():
        kk, nn, pp = k[under], n[under], p[under]
        k1 = kk + 1
        log_p0 = binom.logpmf(k1, nn, pp)
        j = np.arange(n_terms)[None, :]
        kj = k1[:, None] + j
        num = (nn[:, None] - kj) * pp[:, None]
        den = (kj + 1) * (1 - pp[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(num > 0, np.log(num) - np.log(den), -np.inf)
        out[under] = _log_tail_series(log_p0, log_ratio)
    return out.reshape(shape)


_EXACT_ALL_BELOW = 512  # candidate counts up to this are simply evaluated exactly
_NOISE_ALPHA = 0.05  # significance a cut must reach, Bonferroni-corrected, to be worth ranking exactly


def _noise_floor(n_candidates: int) -> float:
    """The log p-value a candidate cut must beat to be worth resolving exactly.

    The threshold search tries every candidate and keeps the best, so the
    best one is a maximum over `n_candidates` tests and needs a Bonferroni
    correction before it means anything. A cut that cannot clear even
    `_NOISE_ALPHA / n_candidates` is noise; which particular noise cut is
    nominally the best changes no decision the caller makes with it.
    """
    return float(np.log(_NOISE_ALPHA) - np.log(max(n_candidates, 1)))


def _fisher_logsf_min_candidates(
    primary_above: np.ndarray, control_above: np.ndarray, n_primary: int, n_control: int, max_exact: int = 256
) -> np.ndarray:
    """`log P(X >= primary_above)` under Fisher's null, per candidate cut, in
    a form whose *argmin* is exact but that never pays the O(draws) exact
    tail for candidates that can't win.

    The exact hypergeometric tail (scipy's C `sf`) sums pmf terms, so a
    threshold search over ~n_seqs candidate cuts costs O(n_seqs^2) -- on
    the GPU it was 85% of `fit`'s wall-clock, the scans themselves 6%. For
    each candidate two O(1) bounds on the tail bracket it instead:
    `pmf(k)` from below (one lgamma expression), and beyond the mode --
    where successive pmf ratios `r` fall below 1 and keep falling --
    `pmf(k) / (1 - r)` from above (a geometric series; below the mode the
    upper bound is just 1). The candidate with the smallest upper bound
    fixes a ceiling `U`; only candidates whose lower bound is <= `U` can
    still be the optimum, and those get the exact tail. Near a real
    optimum `r` is small, the bracket is a few hundredths of a nat wide,
    and a handful of candidates survive (cuts at or below the mode get a
    second lower bound, `1/|support|`, that removes them outright -- see
    the code). Every other candidate is returned as its upper bound
    (never below its true value), so the argmin is the true argmin and
    its value is exact.

    `max_exact` is a batch size, not a cap: the survivors with the
    smallest upper bounds are resolved first, every exact value tightens
    the ceiling (an exact value is itself an upper bound on the minimum),
    and whatever still survives that tighter ceiling is resolved in a
    batch twice as large. Where a cut is genuinely significant this ends
    immediately -- `r` is small, the bracket is tight, and one batch
    resolves everything -- so the argmin is exact and costs nothing.

    The expensive case is the opposite one: no cut is significant, every
    candidate sits near p=1, the bounds can't discriminate, and nearly
    all candidates survive every ceiling. Resolving them exactly is what
    an exact argmin over noise costs, and it is not worth paying -- on a
    7232-sequence run it was 4.8x the whole `fit` (413s -> 86s) and did
    not change a single reported motif. So once a truncated batch has
    failed to bring the ceiling below `_noise_floor` (Bonferroni over the
    candidates searched: a cut that can't clear that bar is noise, and
    its rank among other noise changes nothing downstream), the search
    stops and the remaining candidates keep their upper bounds.

    The guarantee is therefore: **the argmin is exact whenever any cut
    clears the noise floor**, and approximate only among cuts that don't,
    where every returned value is still a valid upper bound -- never
    below the truth, so nothing is reported as more significant than it
    is. With `_EXACT_ALL_BELOW` or fewer candidates everything is
    evaluated exactly, no bounds.
    """
    k = np.asarray(primary_above, dtype=np.int64)
    s = k + np.asarray(control_above, dtype=np.int64)
    M, N = n_primary + n_control, n_primary
    if k.size <= _EXACT_ALL_BELOW:
        return hypergeom_logsf(k - 1, M, s, N)

    lower = hypergeom.logpmf(k, M, s, N)  # vectorized (betaln), O(1) per candidate
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ((s - k) * (N - k)) / ((k + 1) * (M - s - N + k + 1))
        upper = np.where((r >= 0) & (r < 1), lower - np.log1p(-np.clip(r, 0, 1 - 1e-12)), 0.0)
    upper = np.where(np.isfinite(lower), upper, lower)  # pmf(k)=0: impossible cut, keep -inf/nan as is
    lower = np.where(np.isfinite(lower), lower, np.inf)  # ...and never evaluate it exactly
    # A cut at or below the mode is no enrichment at all -- its tail is a
    # large fraction of 1 -- but `pmf(k)` far down the left tail is tiny, so
    # on its own that lower bound keeps every such candidate alive under
    # any ceiling (99% of all exact evaluations on a 7232-sequence run, for
    # cuts that could never win). For `k <= mode` the tail contains
    # `pmf(mode)`, and the largest pmf over a support of `L` points is at
    # least `1/L`. The mode is within 1 of the mean, so `k <= mean - 1`
    # is a safe test for `k <= mode`.
    support = np.minimum(s, N) - np.maximum(0, s + N - M) + 1
    at_or_below_mode = k <= np.floor(N * s / M) - 1
    lower = np.where(at_or_below_mode, np.maximum(lower, -np.log(support)), lower)
    ceiling = float(np.nanmin(upper))
    out = upper.copy()
    resolved = np.zeros(k.size, dtype=bool)
    batch = max_exact
    floor = _noise_floor(k.size)
    while True:
        need = np.nonzero((lower <= ceiling) & ~resolved)[0]
        if need.size == 0:
            return out
        truncated = need.size > batch
        if truncated:
            need = need[np.argsort(upper[need], kind="stable")[:batch]]
            batch *= 2
        out[need] = hypergeom_logsf(k[need] - 1, M, s[need], N)
        resolved[need] = True
        ceiling = min(ceiling, float(np.nanmin(out[need])))
        if truncated and ceiling > floor:
            return out  # nothing significant survives; ranking noise exactly is not worth it


def _log_enrichment_pvalues(primary_above, control_above, n_primary: int, n_control: int) -> np.ndarray:
    """One-sided enrichment log p-value(s) for "primary_above of n_primary vs
    control_above of n_control", vectorized over however many cut points the
    inputs describe: Fisher's exact test (hypergeometric tail), for any
    pair of set sizes.

    STREME's Fisher-vs-Binomial choice is about sequence *lengths* (the
    per-sequence chance of a site differs when lengths differ), not about
    how many sequences each side has; with fixed-width windows lengths are
    always equal, so Fisher is always right here (DESIGNDOC.md §8.2). An
    earlier version switched to a Binomial test against the control rate
    whenever the two sets differed in *size*, which is degenerate when no
    control sequence passes a cut (rate 0, p-value 0 for any primary count
    -- a 19-site 15-mer beat every real motif that way) and had no bounded
    search, so it was also ~100x slower per call.
    """
    primary_above = np.asarray(primary_above)
    control_above = np.asarray(control_above)
    # draw n_primary items (without replacement) from a pool of
    # n_primary+n_control, of which (primary_above+control_above) are
    # "successes" (above threshold); P(X >= primary_above).
    pool = n_primary + n_control
    successes = primary_above + control_above
    return hypergeom_logsf(primary_above - 1, pool, successes, n_primary)


def enrichment_at_threshold(scores_primary, scores_control, threshold: float) -> ThresholdResult:
    """Enrichment test at one *fixed* score cut (score >= `threshold`).

    The hold-out half of STREME's significance story: `optimal_threshold`
    picks the cut on the training sequences (and so its p-value is optimistic
    -- it was selected), then this re-tests that same cut on sequences the
    search never saw, giving `Motif.holdout_logp`. Same test conventions as
    `optimal_threshold` (Fisher's exact test whatever the two set sizes,
    `-inf` scores never count as "above").
    """
    scores_primary = np.asarray(scores_primary, dtype=np.float64).reshape(-1)
    scores_control = np.asarray(scores_control, dtype=np.float64).reshape(-1)
    n_primary, n_control = scores_primary.size, scores_control.size
    if n_primary == 0 or n_control == 0:
        raise ValueError("enrichment_at_threshold requires at least one primary and one control score")
    primary_above = int((scores_primary >= threshold).sum())
    control_above = int((scores_control >= threshold).sum())
    log_p = float(_log_enrichment_pvalues(primary_above, control_above, n_primary, n_control))
    return ThresholdResult(
        threshold=float(threshold),
        log_pvalue=log_p,
        primary_above=primary_above,
        primary_below=n_primary - primary_above,
        control_above=control_above,
        control_below=n_control - control_above,
    )


@dataclass
class CentralEnrichment:
    """Result of `central_enrichment`: `log_pvalue` is Bonferroni-adjusted
    (natural log) over the candidate central-window half-widths tried;
    `half_width` is the winning window (sites with |offset| <= half_width),
    `n_central` how many of the `n_sites` fell inside it, and `expected`
    how many would have under a uniform positional null.
    """

    log_pvalue: float
    half_width: float
    n_central: int
    n_sites: int
    expected: float


def central_enrichment(offsets, n_windows: int) -> CentralEnrichment | None:
    """CentriMo-style central enrichment (§2.7) of best-site positions.

    `offsets` are site *centers* relative to the sequence center, in bases
    (as `Motif.positions` reports them -- half-integers when the motif width
    and sequence length have different parity); `n_windows` is the number of
    possible window starts per sequence (`L - w + 1`), which fixes the null:
    a site is equally likely at any of them, so the chance it lands within
    `h` of center is (number of window starts whose center is within `h`) /
    `n_windows`.

    For every candidate half-width `h` (each distinct |offset| a window
    center can take, out to the sequence edge) the number of sites within
    `h` is tested against that null with a one-sided Binomial test, and the
    best `h` is reported with its log p-value Bonferroni-multiplied by the
    number of candidates tried -- the same max-over-widths-then-correct shape
    CentriMo uses. Returns `None` when there are no sites.
    """
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    n_sites = offsets.size
    if n_sites == 0:
        return None
    if n_windows < 1:
        raise ValueError("central_enrichment requires n_windows >= 1")

    # every window start s in [0, n_windows) has center offset s - (n_windows-1)/2
    window_centers = np.arange(n_windows) - (n_windows - 1) / 2
    candidate_h = np.unique(np.abs(window_centers))
    # all-inclusive h (every window counts as central) can't be enriched; drop it when there's a choice
    if candidate_h.size > 1:
        candidate_h = candidate_h[:-1]

    abs_offsets = np.abs(offsets)
    n_central = (abs_offsets[None, :] <= candidate_h[:, None] + 1e-9).sum(axis=1)
    n_possible = (np.abs(window_centers)[None, :] <= candidate_h[:, None] + 1e-9).sum(axis=1)
    p_null = n_possible / n_windows
    # `binom_logsf`, not scipy's `binom.logsf` (which is just `log(sf)`): a
    # strongly central motif underflows `sf` to 0, and a column of `-inf`
    # would both lose the value and make every underflowing width tie, so
    # `argmin` would pick the narrowest one rather than the best one.
    log_pvals = binom_logsf(n_central - 1, n_sites, p_null) + np.log(candidate_h.size)
    best = int(np.argmin(log_pvals))
    return CentralEnrichment(
        log_pvalue=float(min(log_pvals[best], 0.0)),
        half_width=float(candidate_h[best]),
        n_central=int(n_central[best]),
        n_sites=int(n_sites),
        expected=float(p_null[best] * n_sites),
    )


def optimal_threshold(scores_primary, scores_control) -> ThresholdResult:
    """Sort-once, all-cut-points-at-once threshold optimization (§2.5).

    `scores_primary`/`scores_control` are 1-D best-site scores per sequence
    (e.g. `ScanResult.scores`, one motif's row) -- `-inf` entries (no valid
    window in that sequence, §2.2) are kept as real population members that
    simply never clear any threshold, not dropped.

    Candidate thresholds are every distinct *finite* score value observed in
    either set: any other real cut point would partition the data exactly
    like the nearest such value, so restricting to observed values loses
    nothing and keeps the search over one array instead of a continuum. For
    each candidate `t`, "above" means score >= t.

    Fisher's exact test at every candidate (hypergeometric tail, with the
    bounded search of `_fisher_logsf_min_candidates`), whatever the two set
    sizes -- fixed-width windows make every sequence the same length, which
    is the condition STREME's Fisher branch actually needs (§8.2; see
    `_log_enrichment_pvalues` for why the old size-based Binomial branch is
    gone). One-sided for enrichment (more primary sequences above threshold
    than expected), matching STREME/SEA's convention.

    Returns the `ThresholdResult` for whichever candidate threshold gives the
    smallest (most significant) p-value.
    """
    scores_primary = np.asarray(scores_primary, dtype=np.float64).reshape(-1)
    scores_control = np.asarray(scores_control, dtype=np.float64).reshape(-1)
    n_primary = scores_primary.size
    n_control = scores_control.size
    if n_primary == 0 or n_control == 0:
        raise ValueError("optimal_threshold requires at least one primary and one control score")

    finite_primary = scores_primary[np.isfinite(scores_primary)]
    finite_control = scores_control[np.isfinite(scores_control)]
    candidates = np.unique(np.concatenate([finite_primary, finite_control]))
    if candidates.size == 0:
        raise ValueError("optimal_threshold: no finite scores in either primary or control")

    sorted_primary = np.sort(scores_primary)  # -inf sorts first, never counted "above"
    sorted_control = np.sort(scores_control)

    # count of scores >= t, for every candidate t at once
    primary_above = n_primary - np.searchsorted(sorted_primary, candidates, side="left")
    control_above = n_control - np.searchsorted(sorted_control, candidates, side="left")

    # exact argmin without the exact tail at every cut -- see the helper
    log_pvals = _fisher_logsf_min_candidates(primary_above, control_above, n_primary, n_control)

    best = int(np.argmin(log_pvals))
    return ThresholdResult(
        threshold=float(candidates[best]),
        log_pvalue=float(log_pvals[best]),
        primary_above=int(primary_above[best]),
        primary_below=int(n_primary - primary_above[best]),
        control_above=int(control_above[best]),
        control_below=int(n_control - control_above[best]),
    )
