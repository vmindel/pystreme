"""MotifDiscovery: the public entry point.

See DESIGNDOC.md §3 ("Public API sketch") for the full intended surface.
`scan` (Phase 1), `enrichment` (Phase 2) and `discover` (Phase 3) are the
independently useful pieces; `fit` (Phase 4) is the actual STREME-style
round loop built on top of them: hold-out split, one motif per round,
hold-out significance, erasing, stopping rules, positional distribution.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from . import meme_io
from .control import resolve_control
from .refine import RefinedMotif, column_information, extend_flanks, refine as _refine, trim_flanks
from .scanner import ScanResult, scan as _scan
from .seeds import top_seeds as _top_seeds
from .sequence_store import SequenceStore
from .sites import DEFAULT_P_CUTOFF
from .statistics import (
    CentralEnrichment,
    DecoyCentralEnrichment,
    ThresholdResult,
    central_enrichment,
    decoy_central_enrichment,
    decoy_pwms,
    enrichment_at_threshold,
    optimal_threshold,
)

_ALPHABET = "ACGT"


@dataclass
class Motif:
    """A discovered motif: PWM, consensus, significance, sites, positions.

    Two significance numbers, deliberately kept apart (§5's hold-out
    warning): `log_pvalue` is the enrichment found on the *training*
    sequences -- the same sequences the motif was searched for in, so it's
    an optimistic, selected number, fine for ranking but not for reporting.
    `holdout_logp` re-tests the training threshold on the held-out
    sequences the search never saw; that's the number `fit` uses to decide
    whether a motif counts, and the one to quote. It's `None` when there
    was no hold-out set (`discover()` directly, or `fit(holdout_frac=0)`,
    or too few sequences to split) -- then `log_pvalue` is all there is.

    `sites` covers *every* primary sequence (best score/position/strand,
    with `-inf` where no valid window exists); `n_sites` counts those at or
    above `threshold`. `positions` are the centers of those passing sites
    relative to the sequence center, in bases (§2.7: fixed-width,
    peak-centered windows make this free), and `central` /`central_logp`
    is the CentriMo-style central-enrichment test over them against a
    uniform placement.

    `decoy_central_logp` asks whether the sites are central *beyond what
    the motif's base composition explains*, the question behind "bound
    directly vs. co-occurring". Composition alone pulls a GC-rich motif's
    sites towards GC-rich peak summits, which the uniform null reads as
    central binding. So the sites are also compared with decoys
    (`decoy_central`): column-permuted copies of the PWM, with its
    composition, scanned on the same sequences, whose sites follow the same
    pull. A motif counts as central only if it beats both nulls, uniform
    placement and its decoys, so `decoy_central_logp` is the larger of the
    two p-values (an intersection-union test: valid at its stated level,
    since rejecting needs both). The decoy comparison alone is not enough:
    in GC-cored peaks an AT-leaning motif's decoys are pushed *out* of the
    centre, and uniformly placed sites then look central next to them. The
    decoy comparison by itself is `decoy_central.log_pvalue`. `None` when
    `n_decoys=0` or there are no sites. Decoys of a low-complexity motif
    (a G-run GC box) partly match the motif itself, which makes the test
    conservative, not the reverse.
    """

    pwm: np.ndarray
    consensus: str
    width: int
    seed: str
    threshold: float
    log_pvalue: float
    sites: ScanResult
    n_sites: int
    holdout_logp: float | None = None
    holdout: ThresholdResult | None = field(default=None, repr=False)
    positions: np.ndarray | None = field(default=None, repr=False)
    central_logp: float | None = None
    central: CentralEnrichment | None = field(default=None, repr=False)
    decoy_central_logp: float | None = None
    decoy_central: DecoyCentralEnrichment | None = field(default=None, repr=False)
    round: int = 0
    intervals: list | None = field(default=None, repr=False)  # the extracted windows `sites` is indexed by

    @property
    def name(self) -> str:
        return f"{self.round}-{self.consensus}" if self.round else self.consensus

    def sites_frame(self):
        """Per-peak site table -- see `pystreme.results.sites_frame`."""
        from .results import sites_frame

        return sites_frame(self)

    def __repr__(self) -> str:
        bits = [
            f"width={self.width}",
            f"n_sites={self.n_sites}",
            f"train_logp={self.log_pvalue:.1f}",
        ]
        if self.holdout_logp is not None:
            bits.append(f"holdout_logp={self.holdout_logp:.1f}")
        if self.central_logp is not None:
            bits.append(f"central_logp={self.central_logp:.1f}")
        if self.decoy_central_logp is not None:
            bits.append(f"decoy_central_logp={self.decoy_central_logp:.1f}")
        return f"Motif({self.name}, " + ", ".join(bits) + ")"


def _consensus(pwm: np.ndarray, weak_thresh: float = 0.4) -> str:
    """Per-column argmax base, or `N` where no base clears `weak_thresh`."""
    chars = []
    for col in pwm.T:
        i = int(col.argmax())
        chars.append(_ALPHABET[i] if col[i] >= weak_thresh else "N")
    return "".join(chars)


def _holdout_split(n: int, frac: float, rng: np.random.Generator, min_holdout: int, groups=None):
    """Random (train_idx, holdout_idx) row split; holdout_idx is `None` when
    `frac` is 0 or either side would be smaller than `min_holdout`.

    `groups` (one label per row, numbered 0.. in order of first row, as
    `SequenceStore.holdout_groups` returns) keeps every group whole on one
    side: groups are drawn in random order into the hold-out until it holds
    at least `round(frac * n)` rows. With every row its own group this is
    exactly the plain row permutation, same random draws included.
    """
    n_hold = int(round(frac * n))
    if frac <= 0 or n_hold < min_holdout or n - n_hold < min_holdout:
        return np.arange(n), None
    if groups is None:
        groups = np.arange(n)
    groups = np.asarray(groups)
    n_groups = int(groups.max()) + 1 if groups.size else 0
    perm = rng.permutation(n_groups)
    sizes = np.bincount(groups, minlength=n_groups)
    n_take = int(np.searchsorted(np.cumsum(sizes[perm]), n_hold)) + 1  # first prefix of groups reaching n_hold rows
    in_holdout = np.isin(groups, perm[:n_take])
    hold, train = np.flatnonzero(in_holdout), np.flatnonzero(~in_holdout)
    if hold.size < min_holdout or train.size < min_holdout:
        return np.arange(n), None
    return train, hold


class MotifDiscovery:
    """Fixed-width, GPU-batched de novo motif discovery.

    `from_bed` + `scan` are the Phase 1 scanner; `enrichment` the Phase 2
    SEA-equivalent; `discover` the Phase 3 single-motif finder; `fit` the
    Phase 4 round loop -- the `motifs = disc.fit(idx)` call from §3's API
    sketch. The constructor arguments are the per-object defaults `fit`
    and `discover` fall back to when not given their own.
    """

    def __init__(
        self,
        store: SequenceStore,
        widths=range(6, 16),
        control="dinuc_shuffle",
        holdout_frac: float = 0.1,
        revcomp: bool = True,
        seed: int | None = 0,
    ):
        self.store = store
        self.widths = list(widths)
        self.control = control
        self.holdout_frac = holdout_frac
        self.revcomp = revcomp
        self.seed = seed

    @classmethod
    def from_bed(
        cls,
        peaks,
        genome,
        size: int,
        bg_order: int = 2,
        max_n_frac: float = 0.2,
        device: str = "cpu",
        **kwargs,
    ) -> "MotifDiscovery":
        """Build a store and immediately fit its background (§2.3, order 2 by
        default). `scan()` requires a fitted background, and there's no
        obviously-right *implicit* default for a self-contained discovery
        object the way there might be for a bare scanner call -- see
        `pystreme.scanner.scan`'s docstring for why that stays explicit
        there. Remaining keyword arguments (`widths`, `control`,
        `holdout_frac`, `revcomp`, `seed`) are the constructor's defaults.
        """
        store = SequenceStore.from_bed(peaks, genome, size=size, max_n_frac=max_n_frac, device=device)
        store.fit_background(order=bg_order)
        return cls(store, **kwargs)

    @classmethod
    def from_fasta(cls, path, bg_order: int = 2, max_n_frac: float = 0.2, device: str = "cpu", **kwargs) -> "MotifDiscovery":
        """Equal-length sequences already in a FASTA file (e.g. exported by
        another tool), no genome/BED needed. Same background fitting and
        constructor defaults as `from_bed`."""
        store = SequenceStore.from_fasta(path, device=device, max_n_frac=max_n_frac)
        store.fit_background(order=bg_order)
        return cls(store, **kwargs)

    @classmethod
    def from_sequences(cls, seqs: list[str], bg_order: int = 2, device: str = "cpu", **kwargs) -> "MotifDiscovery":
        """Equal-length sequences in memory (a list of ACGTN strings)."""
        store = SequenceStore.from_sequences(seqs, device=device)
        store.fit_background(order=bg_order)
        return cls(store, **kwargs)

    def scan(
        self,
        motifs,
        idx=None,
        p_cutoff: float = DEFAULT_P_CUTOFF,
        background=None,
        n_jobs: int | None = None,
        overlaps: str | None = "any",
    ):
        """Every site of known or discovered motifs in the peaks, with
        moodspy (MOODS): placements on both strands whose score reaches
        `p_cutoff`, overlapping hits of a motif collapsed to the best one
        (`overlaps=None` keeps them all), so a peak can carry several
        sites or none. See `pystreme.sites.find_sites` for the arguments
        and the table.

        `motifs` is a (4, w) probability PWM, a `Motif` from `fit`, a
        `MemeMotif`, a {name: pwm} dict, or a list of these. Scores are
        log2 odds against the store's order-0 marginals; the best-window
        scorer `fit` uses internally (order-k background, one window per
        peak) is still `pystreme.scanner.scan(pwms, disc.store)`.
        """
        from .sites import find_sites

        return find_sites(
            motifs, self.store, p_cutoff=p_cutoff, background=background, idx=idx, n_jobs=n_jobs, overlaps=overlaps
        )

    def _prepare_primary_and_control(
        self, control, idx, bg_order, n_per_seq, seed=None
    ) -> tuple[SequenceStore, SequenceStore]:
        """Shared setup for `enrichment`/`discover`/`fit`: resolve `idx` into
        a primary store, build the matching control set, and fit *one*
        background model on the control, applied to both (§2.3's "fit on the
        control set" recommendation -- fitting on the primary being tested
        would bias the background towards it).

        Control *index arrays* are resolved against the whole `self.store`
        (`universe=`), which is what the public docstrings promise, while
        generated controls are built to match the `idx` subset actually
        being tested -- the two only differ when `idx` was given.
        """
        # a view, not `self.store` itself: the background applied below must
        # not leak into later `disc.scan(...)` calls or `to_meme`
        primary_store = self.store.view() if idx is None else self.store.subset(idx)
        order = bg_order if bg_order is not None else primary_store.bg_order
        if order is None:
            raise RuntimeError(
                "MotifDiscovery: no bg_order available -- pass bg_order= explicitly, "
                "or fit one first (from_bed does this automatically)"
            )
        control_store = resolve_control(control, primary_store, n_per_seq=n_per_seq, seed=seed, universe=self.store)
        if control_store.length != primary_store.length:
            # Every test here compares per-sequence best scores, which is
            # only a fair comparison when both sides offer the same number
            # of windows to win from (§8.2: equal lengths are exactly the
            # condition that makes Fisher the right test). Unequal lengths
            # would silently bias enrichment towards whichever side is
            # longer, so they're rejected rather than half-corrected.
            raise ValueError(
                f"control sequences are {control_store.length} bp but primary sequences are "
                f"{primary_store.length} bp: pystreme's fixed-width windows require equal lengths "
                "on both sides (same number of possible motif starts per sequence). Re-extract the "
                "control regions at the primary width."
            )
        bg = control_store.background_model(order)  # estimated once, applied to both
        control_store.apply_background(bg)
        primary_store.apply_background(bg)
        return primary_store, control_store

    def enrichment(
        self,
        pwm,
        control=None,
        idx=None,
        revcomp: bool | None = None,
        kernel: str = "auto",
        n_per_seq: int = 1,
        bg_order: int | None = None,
        seed: int | None = None,
    ) -> ThresholdResult:
        """SEA-equivalent enrichment test for one known motif (§"Phase 2"
        deliverable): scan `pwm` against the primary sequences and against a
        matched control set, then find the score threshold that most
        significantly enriches primary over control (§2.5).

        `control` is `"dinuc_shuffle"` (the constructor default), a
        `control.DinucShuffle`/`control.GCMatched`/`control.Explicit`
        instance, or anything `control.Explicit` accepts directly -- a
        `SequenceStore`, `list[str]`, or an index array into `self.store`
        (see `control.resolve_control`). `idx` restricts which of
        `self.store`'s sequences count as "primary" (the control set is then
        built to match that subset, not the whole store). `bg_order`
        defaults to whatever `from_bed` already fit (`self.store.bg_order`).
        """
        control = self.control if control is None else control
        revcomp = self.revcomp if revcomp is None else revcomp
        seed = self.seed if seed is None else seed
        primary_store, control_store = self._prepare_primary_and_control(control, idx, bg_order, n_per_seq, seed)
        primary_scores = _scan([pwm], primary_store, revcomp=revcomp, kernel=kernel).scores[0]
        control_scores = _scan([pwm], control_store, revcomp=revcomp, kernel=kernel).scores[0]
        return optimal_threshold(primary_scores.cpu().numpy(), control_scores.cpu().numpy())

    # ------------------------------------------------------------------
    # One round: seeds -> refinement -> best motif across widths
    # ------------------------------------------------------------------

    @staticmethod
    def _rescore(pwm: np.ndarray, seed: str, primary_store, control_store, revcomp: bool, kernel: str) -> RefinedMotif:
        """One scan + threshold search for a given PWM (no re-estimation):
        the training-set numbers a trimmed motif needs before it can be
        compared with the others."""
        p = _scan([pwm], primary_store, revcomp=revcomp, kernel=kernel).scores[0]
        c = _scan([pwm], control_store, revcomp=revcomp, kernel=kernel).scores[0]
        thr = optimal_threshold(p.cpu().numpy(), c.cpu().numpy())
        n_sites = int(((p >= thr.threshold) & torch.isfinite(p)).sum())
        return RefinedMotif(
            pwm=pwm, seed=seed, width=pwm.shape[1], threshold=thr.threshold, log_pvalue=thr.log_pvalue, n_sites=n_sites
        )

    @classmethod
    def _discover_on(
        cls,
        primary_store: SequenceStore,
        control_store: SequenceStore,
        widths,
        n_per_width: int,
        n_iter: int,
        revcomp: bool,
        kernel: str,
        min_flank_ic: float = 0.3,
        width_tolerance: float = 2.0,
    ) -> RefinedMotif | None:
        """Find seed words per width (§2.1), refine each width's best seeds,
        and return the best refined motif across widths, or `None` if no
        width had any seed word at all.

        Widths are compared by their per-*sequence* Fisher/Binomial p-value
        on the training set (how many sequences have a site clearing the
        enrichment-optimal threshold) -- STREME's own objective. That
        objective saturates: once a motif separates primary from control
        about as well as the data allows, a version one column shorter or
        longer lands within noise of the same p-value (a shorter core
        typically picks up a couple of extra partial matches, a longer one
        carries a near-flat flank column the score barely sees), so the
        p-value alone can't fix the width. Two additions handle that:

        - flank columns with information below `min_flank_ic` bits are
          trimmed (never below the smallest requested width), then the
          motif is grown back outward while the next column beyond an edge
          carries at least `min_flank_ic` bits (never above the largest
          requested width) -- a seed one base off the real site otherwise
          refines into a motif missing its first or last column -- and a
          motif whose width changed is re-scored;
        - candidates whose log p-value is within `width_tolerance` (natural
          log units) of the best are treated as tied, and the tie goes to
          the largest total information content -- the most specific
          description of the site the data supports equally well.

        The hold-out re-test in `fit` is what keeps the training-set
        selection here from turning into an overstated significance.
        """
        widths = list(widths)
        min_width, max_width = (min(widths), max(widths)) if widths else (1, 1)
        seeds_by_width = _top_seeds(primary_store, control_store, widths, n_per_width=n_per_width, revcomp=revcomp)

        candidates: list[RefinedMotif] = []
        for w, seeds in seeds_by_width.items():
            if not seeds:
                continue
            result = _refine(
                [s.word for s in seeds], primary_store, control_store, n_iter=n_iter, revcomp=revcomp, kernel=kernel
            )
            pwm, _, _ = trim_flanks(result.pwm, min_flank_ic, min_width=min_width)
            pwm = extend_flanks(
                pwm, result.threshold, primary_store, min_flank_ic, max_width, revcomp, kernel,
                control_store=control_store,
            )
            # Any change to the matrix needs re-scoring, not just a change of
            # width: trimming one uninformative flank and growing an
            # informative one on the other side lands back on the original
            # width with a different PWM, and the correction used to be
            # dropped on the floor in exactly that case.
            if pwm.shape != result.pwm.shape or not np.array_equal(pwm, result.pwm):
                result = cls._rescore(pwm, result.seed, primary_store, control_store, revcomp, kernel)
            candidates.append(result)
        if not candidates:
            return None

        best_logp = min(c.log_pvalue for c in candidates)
        tied = [c for c in candidates if c.log_pvalue <= best_logp + width_tolerance]
        return max(tied, key=lambda c: (float(column_information(c.pwm).sum()), -c.log_pvalue))

    def _finalize(
        self,
        best: RefinedMotif,
        primary_full: SequenceStore,
        holdout: ThresholdResult | None,
        round_i: int,
        revcomp: bool,
        kernel: str,
        n_decoys: int = 10,
        seed: int | None = None,
    ) -> Motif:
        """Turn a refined PWM into a reported `Motif`: sites on *all* primary
        sequences (train and hold-out, nothing erased), passing-site
        positions relative to center, central enrichment (§2.7) against a
        uniform placement and against `n_decoys` composition-matched decoys.

        The decoys are scanned in the same call as the motif, so under the
        same background. Each contributes its best site in the `n_sites`
        sequences where it scores highest -- selected as strongly as the
        motif's own sites, rather than at the motif's threshold, which means
        something different on a different matrix.
        """
        decoys = decoy_pwms(best.pwm, n_decoys, np.random.default_rng(None if seed is None else [seed, round_i]))
        res = _scan([best.pwm, *decoys], primary_full, revcomp=revcomp, kernel=kernel)
        scores, positions, strands = res.scores[0], res.positions[0], res.strands[0]
        passing = (scores >= best.threshold) & torch.isfinite(scores)
        w, length = best.width, primary_full.length
        to_offset = (w - 1) / 2 - (length - 1) / 2
        offsets = positions[passing].cpu().numpy().astype(np.float64) + to_offset
        central = central_enrichment(offsets, n_windows=length - w + 1) if length >= w else None

        decoy_central = None
        n_sites = int(passing.sum())
        if decoys and n_sites and length >= w:
            d_scores = res.scores[1:].cpu().numpy()
            d_positions = res.positions[1:].cpu().numpy()
            decoy_offsets = []
            for k in range(len(decoys)):
                finite = np.flatnonzero(np.isfinite(d_scores[k]))
                top = finite[np.argsort(-d_scores[k][finite], kind="stable")[:n_sites]]
                decoy_offsets.append(d_positions[k][top].astype(np.float64) + to_offset)
            decoy_central = decoy_central_enrichment(
                offsets, np.concatenate(decoy_offsets), n_windows=length - w + 1, n_decoys=len(decoys)
            )
        return Motif(
            pwm=best.pwm,
            consensus=_consensus(best.pwm),
            width=w,
            seed=best.seed,
            threshold=best.threshold,
            log_pvalue=best.log_pvalue,
            sites=ScanResult(scores, positions, strands),
            n_sites=int(passing.sum()),
            holdout_logp=None if holdout is None else holdout.log_pvalue,
            holdout=holdout,
            positions=offsets,
            central_logp=None if central is None else central.log_pvalue,
            central=central,
            # central against uniform placement AND against the decoys
            decoy_central_logp=None
            if decoy_central is None or central is None
            else max(central.log_pvalue, decoy_central.log_pvalue),
            decoy_central=decoy_central,
            round=round_i,
            intervals=list(primary_full.intervals),
        )

    @staticmethod
    def _erase_motif(
        stores: list[SequenceStore], best: RefinedMotif, revcomp: bool, kernel: str, max_passes: int = 10
    ) -> int:
        """STREME's erasing step: in every store, mask out each sequence's
        best site while it still clears the motif's threshold, so the next
        round can't rediscover this motif. Repeated (up to `max_passes`)
        because the scanner reports one best site per sequence and a
        sequence may carry several. Scanning between passes reuses the
        pre-erase background cumsum: erased windows are masked anyway, so
        the stale values are never consulted for a window that can still
        win. Callers refit the background properly before the next round.
        Returns the total number of windows erased.
        """
        total = 0
        for store in stores:
            cumsum = store.bg_cumsum
            for _ in range(max_passes):
                store.bg_cumsum = cumsum
                res = _scan([best.pwm], store, revcomp=revcomp, kernel=kernel)
                scores, positions = res.scores[0], res.positions[0]
                passing = (scores >= best.threshold) & torch.isfinite(scores)
                if not bool(passing.any()):
                    break
                rows = torch.nonzero(passing).squeeze(1)
                total += store.erase(rows, positions[rows], best.width)
            store.bg_cumsum = None  # stale after erasing; the round loop refits
        return total

    # ------------------------------------------------------------------
    # Public discovery entry points
    # ------------------------------------------------------------------

    def discover(
        self,
        widths=None,
        control=None,
        idx=None,
        n_per_width: int = 4,
        n_iter: int = 20,
        revcomp: bool | None = None,
        kernel: str = "auto",
        n_per_seq: int = 1,
        bg_order: int | None = None,
        seed: int | None = None,
        min_flank_ic: float = 0.3,
        width_tolerance: float = 2.0,
        n_decoys: int = 10,
    ) -> Motif:
        """Single-motif de novo discovery (§"Phase 3" deliverable): find seed
        words per width (§2.1), refine each width's best seeds (§2.1's
        batched-convolution refinement loop), and return the single best
        motif across all widths (see `_discover_on` for how widths are
        compared: `min_flank_ic` trims flat flank columns, `width_tolerance`
        breaks near-ties by information content).

        This is one round of `fit` without the hold-out split, erasing, or
        stopping rules -- one motif, one significance number computed on
        the same data it was found in (`Motif.holdout_logp` stays `None`;
        see `Motif`'s docstring). Positions and both central-enrichment
        tests (uniform and decoy, `n_decoys`) are filled in the same way
        `fit` does.
        """
        widths = self.widths if widths is None else widths
        control = self.control if control is None else control
        revcomp = self.revcomp if revcomp is None else revcomp
        seed = self.seed if seed is None else seed
        primary_store, control_store = self._prepare_primary_and_control(control, idx, bg_order, n_per_seq, seed)
        best = self._discover_on(
            primary_store, control_store, widths, n_per_width, n_iter, revcomp, kernel, min_flank_ic, width_tolerance
        )
        if best is None:
            raise RuntimeError("MotifDiscovery.discover: no seed words found for any requested width")
        return self._finalize(
            best, primary_store, holdout=None, round_i=0, revcomp=revcomp, kernel=kernel, n_decoys=n_decoys, seed=seed
        )

    def fit(
        self,
        primary_idx=None,
        n_motifs: int = 3,
        pvalue_thresh: float = 0.05,
        *,
        widths=None,
        control=None,
        holdout_frac: float | None = None,
        seed: int | None = None,
        n_per_width: int = 4,
        n_iter: int = 20,
        revcomp: bool | None = None,
        kernel: str = "auto",
        n_per_seq: int = 1,
        bg_order: int | None = None,
        patience: int = 3,
        keep_insignificant: bool = False,
        min_holdout: int = 10,
        min_flank_ic: float = 0.3,
        width_tolerance: float = 2.0,
        n_decoys: int = 10,
        verbose: bool = False,
    ) -> list[Motif]:
        """The round loop (§"Phase 4" deliverable): `motifs = disc.fit(idx)`.

        1. Resolve `primary_idx` (all of `self.store` by default) and build
           the control set (`control`, seeded by `seed`); fit the background
           on the control (§2.3).
        2. Split primary and control into training / hold-out subsets
           (`holdout_frac`, seeded). Hold-out is skipped, with a warning,
           when either side would have fewer than `min_holdout` sequences.
        3. Each round: seeds -> refinement -> best motif on the training
           set (`discover`'s machinery); re-test its threshold on the
           hold-out set (`Motif.holdout_logp`); report sites/positions/
           central enrichment on the full primary set; then erase the
           motif's sites from every working set and go again.
        4. Stop after `n_motifs` significant motifs (`holdout_logp <=
           log(pvalue_thresh)`, or the training p-value when there is no
           hold-out), or after `patience` consecutive insignificant ones,
           or when no seed words remain.

        Insignificant motifs are dropped unless `keep_insignificant` (they
        are still erased before continuing, as in STREME). Motifs come back
        in discovery order, `Motif.round` numbered from 1. `min_flank_ic`
        and `width_tolerance` control how the per-round width is settled
        (see `_discover_on`). `n_decoys` composition-matched decoy matrices
        per motif give `Motif.decoy_central_logp` (0 skips it); they are
        drawn from `seed` and the round number, so the same call gives the
        same decoys.

        Hold-out caveat (§5): the hold-out p-value is valid for *one* call.
        Calling `fit` repeatedly over overlapping subsets and picking the
        best result reintroduces the selection the split was meant to
        remove -- vary `seed` per call and keep a never-touched set for a
        final check if that's the workflow.
        """
        widths = list(self.widths if widths is None else widths)
        control = self.control if control is None else control
        holdout_frac = self.holdout_frac if holdout_frac is None else holdout_frac
        seed = self.seed if seed is None else seed
        revcomp = self.revcomp if revcomp is None else revcomp
        if n_motifs < 1:
            raise ValueError("fit: n_motifs must be >= 1")
        if not (0 < pvalue_thresh <= 1):
            raise ValueError("fit: pvalue_thresh must be in (0, 1]")

        primary_full, control_full = self._prepare_primary_and_control(control, primary_idx, bg_order, n_per_seq, seed)
        order = primary_full.bg_order

        rng = np.random.default_rng(seed)
        # overlapping or identical primary sequences stay on one side of the
        # split; control rows are shuffles or independent draws, so they don't need it
        p_train_idx, p_hold_idx = _holdout_split(
            primary_full.n_seqs, holdout_frac, rng, min_holdout, groups=primary_full.holdout_groups()
        )
        c_train_idx, c_hold_idx = _holdout_split(control_full.n_seqs, holdout_frac, rng, min_holdout)
        use_holdout = p_hold_idx is not None and c_hold_idx is not None
        if holdout_frac > 0 and not use_holdout:
            warnings.warn(
                f"fit: hold-out disabled -- holdout_frac={holdout_frac} of {primary_full.n_seqs} primary / "
                f"{control_full.n_seqs} control sequences leaves fewer than min_holdout={min_holdout} on one side "
                "(overlapping or identical primary sequences are kept together, which can make a split impossible); "
                "significance is the (optimistic) training-set p-value",
                stacklevel=2,
            )
            p_train_idx, c_train_idx = np.arange(primary_full.n_seqs), np.arange(control_full.n_seqs)

        p_train, c_train = primary_full.subset(p_train_idx), control_full.subset(c_train_idx)
        working = [c_train, p_train]
        if use_holdout:
            p_hold, c_hold = primary_full.subset(p_hold_idx), control_full.subset(c_hold_idx)
            working += [c_hold, p_hold]

        log_thresh = math.log(pvalue_thresh)
        motifs: list[Motif] = []
        n_significant = 0
        n_failed = 0
        round_i = 0
        while True:
            round_i += 1
            # One background per round, fitted on the (progressively erased)
            # training control and applied to every store this round scores
            # -- including `primary_full`, which `_finalize` reports sites
            # from. It used to keep the setup background (fitted on the
            # *whole*, unerased control), so the threshold learned on
            # `p_train` under one scoring model was applied to scores from
            # another: n_sites, positions, central enrichment and the BED
            # export could all disagree with the motif's own threshold, and
            # a perfectly good motif could report zero sites. `primary_full`
            # is refit, never erased -- reporting is on whole sequences.
            bg = c_train.background_model(order)
            for s in working + [primary_full]:
                s.apply_background(bg)

            best = self._discover_on(
                p_train, c_train, widths, n_per_width, n_iter, revcomp, kernel, min_flank_ic, width_tolerance
            )
            if best is None:
                if verbose:
                    print(f"round {round_i}: no seed words left; stopping")
                break

            holdout = None
            if use_holdout:
                hp = _scan([best.pwm], p_hold, revcomp=revcomp, kernel=kernel).scores[0].cpu().numpy()
                hc = _scan([best.pwm], c_hold, revcomp=revcomp, kernel=kernel).scores[0].cpu().numpy()
                holdout = enrichment_at_threshold(hp, hc, best.threshold)
            motif = self._finalize(best, primary_full, holdout, round_i, revcomp, kernel, n_decoys=n_decoys, seed=seed)
            logp = motif.holdout_logp if use_holdout else motif.log_pvalue
            significant = logp <= log_thresh
            if verbose:
                print(f"round {round_i}: {motif!r} {'accepted' if significant else 'rejected'}")

            if significant or keep_insignificant:
                motifs.append(motif)
            if significant:
                n_significant += 1
                n_failed = 0
            else:
                n_failed += 1
            if n_significant >= n_motifs or n_failed >= patience:
                break
            if self._erase_motif(working, best, revcomp, kernel) == 0:
                break  # nothing to erase means the next round would find this again
        return motifs

    def to_bed(
        self,
        motifs,
        path,
        p_cutoff: float = DEFAULT_P_CUTOFF,
        background=None,
        n_jobs: int | None = None,
        overlaps: str | None = "any",
    ) -> int:
        """Write every moodspy site of `motifs` (see `scan`) as BED6:
        genomic coordinates, `motif|peak` name, log2-odds score, strand.
        Returns the number of sites written. `pystreme.results.sites_bed`
        still writes `fit`'s one best window per peak instead."""
        from .sites import write_sites_bed

        sites = self.scan(motifs, p_cutoff=p_cutoff, background=background, n_jobs=n_jobs, overlaps=overlaps)
        return write_sites_bed(sites, path)

    def to_meme(self, motifs, path, background: np.ndarray | None = None):
        """Write motifs to MEME minimal format. See `pystreme.meme_io.write_meme`.

        `motifs` is a list of `Motif` (as `fit` returns; named
        `<round>-<consensus>`, with `nsites` and the hold-out p-value -- or
        the training one when there is none -- in the `E=` slot), or the
        {name: (4, w) array} / list-of-`MemeMotif` forms `write_meme`
        accepts directly.
        """
        if isinstance(motifs, Motif):
            motifs = [motifs]
        if isinstance(motifs, (list, tuple)) and motifs and isinstance(motifs[0], Motif):
            items = []
            for i, m in enumerate(motifs):
                logp = m.holdout_logp if m.holdout_logp is not None else m.log_pvalue
                name = m.name if m.round else f"{i + 1}-{m.consensus}"
                items.append(
                    meme_io.MemeMotif(name=name, pwm=m.pwm, nsites=m.n_sites, evalue=math.exp(logp), log_evalue=logp)
                )
            motifs = items
        store = getattr(self, "store", None)
        if background is None and store is not None and store.bg_marginals is not None:
            # MEME's header takes one row of base frequencies; an order-k
            # store's conditional table has no such row, but its order-0
            # marginals (kept whatever the order, and used at sequence
            # starts) are exactly that row. Before, every default order-2
            # run wrote a MEME file with no background at all.
            background = store.bg_marginals.exp().cpu().numpy()
        meme_io.write_meme(motifs, path, background=background)
