"""PWM refinement: the iterative loop that dominates STREME's runtime.

See DESIGNDOC.md §2.1 (best-site lookup as batched convolution -- the reason
this is fast here) and §3 (Refiner). Phase 3.

Each iteration: scan primary+control with the current PWM, find the
enrichment-optimal score threshold (§2.5), then re-estimate the PWM from the
primary sites clearing that threshold (their actual best-matching window, on
whichever strand won). This is the same NREF-seeds x NITER-iterations
structure §2.1 costs out for STREME -- a handful of seeds per width, ~20
iterations each, each iteration one batched GPU scan rather than a suffix
tree walk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .scanner import scan
from .sequence_store import SequenceStore
from .statistics import optimal_threshold

_ALPHABET = "ACGT"


@dataclass
class RefinedMotif:
    pwm: np.ndarray
    seed: str
    width: int
    threshold: float
    log_pvalue: float
    n_sites: int


def seed_to_pwm(word: str, sharpness: float = 0.8) -> np.ndarray:
    """A discrete seed word -> a (4, w) PWM: `sharpness` probability at the
    observed base, the remainder split evenly across the other three --
    concentrated enough to scan meaningfully, soft enough that refinement
    (rather than the seed itself) does the real work.
    """
    w = len(word)
    pwm = np.full((4, w), (1 - sharpness) / 3)
    for i, c in enumerate(word):
        pwm[_ALPHABET.index(c), i] = sharpness
    return pwm / pwm.sum(axis=0, keepdims=True)


def column_information(pwm: np.ndarray) -> np.ndarray:
    """Per-column information content in bits, relative to a uniform base
    distribution: `sum_b p log2(4 p)`, 0 for a flat column, 2 for a fixed base."""
    pwm = np.clip(np.asarray(pwm, dtype=np.float64), 1e-12, 1.0)
    pwm = pwm / pwm.sum(axis=0, keepdims=True)
    return (pwm * np.log2(4 * pwm)).sum(axis=0)


def trim_flanks(pwm: np.ndarray, min_ic: float, min_width: int = 1) -> tuple[np.ndarray, int, int]:
    """Drop leading/trailing columns with information below `min_ic` bits,
    never going below `min_width` columns. Returns `(trimmed_pwm, n_left,
    n_right)` -- the counts trimmed from each end, so a site position for
    the original PWM can be shifted by `n_left` to point at the trimmed one.

    A refined PWM one or two columns wider than the real site keeps its
    extra columns near-flat (they were estimated from whatever base happens
    to flank each site), and the enrichment objective is blind to them --
    see `MotifDiscovery._discover_on`. Trimming them is how the reported
    width ends up meaning something.
    """
    ic = column_information(pwm)
    left, right = 0, pwm.shape[1]
    while right - left > min_width and ic[left] < min_ic:
        left += 1
    while right - left > min_width and ic[right - 1] < min_ic:
        right -= 1
    return pwm[:, left:right], left, pwm.shape[1] - right


def extend_flanks(
    pwm: np.ndarray,
    threshold: float,
    store: SequenceStore,
    min_ic: float,
    max_width: int,
    revcomp: bool = True,
    kernel: str = "auto",
    control_store: SequenceStore | None = None,
) -> np.ndarray:
    """Grow a refined PWM outward while the column just beyond either edge
    is informative: re-estimate a (4, w+2) matrix from the sequences whose
    best site clears `threshold` (their windows widened by one base on each
    side, in motif orientation), keep an edge column if its information is
    at least `min_ic` bits, and repeat until neither edge qualifies or
    `max_width` is reached.

    The mirror image of `trim_flanks`: a seed word that sits one base off
    the real site refines into a motif missing a column at one end (the
    fixed-width refinement can't shift its window), and the extra column
    at the other end is flat. Trimming removes the flat one, this puts the
    missing one back. Sites too close to a sequence edge to widen are
    simply left out of the re-estimate. The caller re-scores the result
    (its threshold changes with width).

    `threshold` is only the starting point: score scale moves with width,
    so a threshold fitted to the pre-trim PWM is the wrong cut for the
    trimmed one (systematically too strict -- fewer columns, lower scores)
    and for each widened one in turn. Given `control_store`, the
    enrichment-optimal threshold is refitted for the current PWM at every
    step, so sites are selected by a cut that belongs to the matrix doing
    the selecting. Without it the passed threshold is used throughout, as
    before.
    """
    while pwm.shape[1] < max_width:
        w = pwm.shape[1]
        res = scan([pwm], store, revcomp=revcomp, kernel=kernel)
        scores, positions, strands = res.scores[0], res.positions[0], res.strands[0]
        if control_store is not None:
            c_scores = scan([pwm], control_store, revcomp=revcomp, kernel=kernel).scores[0]
            threshold = optimal_threshold(scores.cpu().numpy(), c_scores.cpu().numpy()).threshold
        passing = (scores >= threshold) & torch.isfinite(scores) & (positions >= 1) & (positions + w + 1 <= store.length)
        wider = _aligned_site_counts(store, positions - 1, strands, passing, w + 2)
        if wider is None:
            break
        ic = column_information(wider)
        grow_left = ic[0] >= min_ic
        grow_right = ic[-1] >= min_ic and (not grow_left or w + 2 <= max_width)
        if not (grow_left or grow_right):
            break
        left = 0 if grow_left else 1
        right = wider.shape[1] if grow_right else wider.shape[1] - 1
        pwm = wider[:, left:right]
    return pwm


def _aligned_site_counts(
    store: SequenceStore, positions: torch.Tensor, strands: torch.Tensor, passing: torch.Tensor, width: int
) -> np.ndarray | None:
    """Re-estimate a (4, w) PWM from the primary sites in `passing`: each
    such sequence's actual best-matching window, reverse-complemented back
    to the strand the PWM itself is on when `strands` says RC won. Returns
    `None` if no sites pass (nothing to re-estimate from).
    """
    if int(passing.sum()) == 0:
        return None
    device = store.codes.device
    rows = torch.nonzero(passing).squeeze(1)
    positions, strands = positions[rows], strands[rows]  # only passing rows are indexed at all
    offsets = torch.arange(width, device=device)
    win_idx = positions.unsqueeze(1) + offsets.unsqueeze(0)  # (n_passing, width)
    windows = torch.gather(store.codes[rows].long(), 1, win_idx)  # forward-strand codes
    rc = (3 - windows).flip(dims=[1])
    windows = torch.where((strands < 0).unsqueeze(1), rc, windows)  # (n_passing, width)

    counts = torch.stack([(windows == b).sum(dim=0) for b in range(4)]).float()  # (4, width)
    counts += 0.1  # pseudocount: keep every column strictly positive for log-odds later
    pwm = counts / counts.sum(dim=0, keepdim=True)
    return pwm.cpu().numpy()


def refine(
    seed_words: list[str],
    primary_store: SequenceStore,
    control_store: SequenceStore,
    n_iter: int = 20,
    revcomp: bool = True,
    kernel: str = "auto",
) -> RefinedMotif:
    """Refine a batch of same-width seed words; return the single best result.

    Both stores must already have a fitted background of the same order
    (§2.3's "fit on the control set" -- see `MotifDiscovery`'s callers of
    this function for how that's set up).
    """
    if not seed_words:
        raise ValueError("refine() requires at least one seed word")
    width = len(seed_words[0])
    if any(len(w) != width for w in seed_words):
        raise ValueError("refine() requires all seed words to be the same width")

    # All seeds of the width iterate together: one batched scan of the
    # primary store and one of the control per iteration (§2.1's "batched
    # per width" -- the scanner stacks same-width PWMs into a single kernel
    # call), instead of one pair of scans per seed per iteration. Seeds
    # drop out of the batch as they converge or die.
    pwms = [seed_to_pwm(word) for word in seed_words]
    active = list(range(len(seed_words)))
    best: RefinedMotif | None = None
    best_ic = -1.0
    for _ in range(n_iter):
        if not active:
            break
        batch = [pwms[i] for i in active]
        p_result = scan(batch, primary_store, revcomp=revcomp, kernel=kernel)
        c_result = scan(batch, control_store, revcomp=revcomp, kernel=kernel)
        p_scores_all = p_result.scores.cpu().numpy()  # one device sync per store per iteration
        c_scores_all = c_result.scores.cpu().numpy()

        still_active = []
        for j, i in enumerate(active):
            pwm = pwms[i]
            thr = optimal_threshold(p_scores_all[j], c_scores_all[j])
            passing_np = (p_scores_all[j] >= thr.threshold) & np.isfinite(p_scores_all[j])
            n_sites = int(passing_np.sum())

            # Ties on p-value go to the PWM with more information: once the
            # objective saturates (see MotifDiscovery._discover_on) the
            # data-estimated PWMs tie the soft 0.8-sharpness seed exactly (and
            # must beat it, or the seed itself gets reported), and a PWM that
            # has drifted one column off the site (one flat flank, one real
            # column lost) ties the aligned one and must lose to it.
            ic = float(column_information(pwm).sum())
            if best is None or thr.log_pvalue < best.log_pvalue or (thr.log_pvalue == best.log_pvalue and ic > best_ic):
                best = RefinedMotif(
                    pwm=pwm, seed=seed_words[i], width=width, threshold=thr.threshold, log_pvalue=thr.log_pvalue,
                    n_sites=n_sites,
                )
                best_ic = ic

            passing = torch.from_numpy(passing_np).to(primary_store.codes.device)
            next_pwm = _aligned_site_counts(primary_store, p_result.positions[j], p_result.strands[j], passing, width)
            if next_pwm is None:
                continue  # no sites clear the threshold; nothing left to refine from
            if np.allclose(next_pwm, pwm, atol=1e-6):
                continue  # converged: the same sites re-estimate the same PWM, so nothing changes from here
            pwms[i] = next_pwm
            still_active.append(i)
        active = still_active
    return best
