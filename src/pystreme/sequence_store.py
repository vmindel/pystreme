"""SequenceStore: the fixed-width, GPU-resident sequence universe.

See DESIGNDOC.md §2.2 (fixed-width windows), §2.3 (higher-order background via
cumsum), and §3 (architecture) for the design this implements. Phase 1.

Layout, per §3:
    codes       (N, L)     uint8, 0-3 (ACGT) + 4 for N/other IUPAC ambiguity
    mask        (N, L)     bool, True where the base is a real call (not N)
    bg_cumsum   (N, L+1)   fp32, cumulative sum of the order-k background
                            log-likelihood, set once fit_background() is called
    onehot()                (N, 4, L) fp16 by default, computed lazily and cached
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

# code 0-3 = A,C,G,T ; 4 = N / any other IUPAC ambiguity code
_ALPHABET = "ACGT"
_N_CODE = 4
_BLOCK_GAP = 1_000_000  # from_bed: windows on one chromosome closer than this share one genome read

_LUT = np.full(256, _N_CODE, dtype=np.uint8)
for _i, _b in enumerate(_ALPHABET):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i  # soft-masked (repeat) bases are still real bases


def encode(seq: str) -> np.ndarray:
    """ASCII DNA string -> uint8 codes (0-3 ACGT, 4 = N/other), vectorized."""
    return _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


@dataclass
class Interval:
    chrom: str
    start: int
    end: int
    name: str = ""


@dataclass
class BackgroundModel:
    """An estimated order-k Markov background: the tables, not the scores.

    `tables[j]` is the order-j conditional model, shape (4**j, 4), in
    natural log; `tables[0]` is the order-0 marginal row every higher order
    backs off to. Estimated by `SequenceStore.background_model` and applied
    -- to any number of stores -- by `SequenceStore.apply_background`.
    `n_seqs`/`length` record what it was fitted on, so a model that has
    travelled between stores can still say where it came from.
    """

    order: int
    tables: list  # index j -> (4**j, 4) log conditionals; tables[0] is (4,)
    pseudocount: float = 1.0
    n_seqs: int = 0
    length: int = 0

    @property
    def marginals(self):
        """The order-0 log base frequencies (what `to_meme` writes)."""
        return self.tables[0]


@dataclass
class Exclusion:
    interval: Interval
    reason: str  # "chrom_edge" | "unknown_chrom" | "max_n_frac"


def _parse_bed(path) -> list[Interval]:
    intervals = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                fields = line.split()
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            name = fields[3] if len(fields) > 3 else f"peak_{i}"
            intervals.append(Interval(chrom, start, end, name))
    return intervals


class SequenceStore:
    """Fixed-width (N, L) sequence universe with an order-k background model.

    Construct via `from_bed` or `from_fasta`. See DESIGNDOC.md §2.2 for the
    extraction and edge-handling policy this follows.
    """

    def __init__(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor,
        intervals: list[Interval],
        excluded: list[Exclusion],
        device: str = "cpu",
    ):
        assert codes.shape == mask.shape
        assert codes.shape[0] == len(intervals)
        self.codes = codes
        self.mask = mask
        self.intervals = intervals
        self.excluded = excluded
        self.device = device
        self.n_seqs, self.length = codes.shape
        self._onehot_cache: dict[torch.dtype, torch.Tensor] = {}
        self.bg_cumsum: torch.Tensor | None = None
        self.bg_table: torch.Tensor | None = None
        self.bg_marginals: torch.Tensor | None = None  # order-0 log frequencies, whatever bg_order is
        self.bg_tables: list[torch.Tensor] | None = None  # index j -> order-j model, for the graded fallback
        self.bg_model: BackgroundModel | None = None  # the estimated model bg_cumsum came from
        self.bg_order: int | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_bed(
        cls,
        peaks,
        genome,
        size: int,
        max_n_frac: float = 0.2,
        device: str = "cpu",
    ) -> "SequenceStore":
        """Fixed-width extraction: `center ± size/2`, HOMER `-size` semantics.

        `peaks` is a BED path (or an already-parsed list of Interval).
        `genome` is a FASTA path (or an open pyfaidx.Fasta) with a `.fai`
        index alongside it. Sequences whose window would run off a
        chromosome end, or reference a chromosome not in the genome, are
        dropped with a warning; so are sequences whose N-fraction exceeds
        `max_n_frac`. See DESIGNDOC.md §2.2.
        """
        import pyfaidx

        raw_intervals = peaks if isinstance(peaks, list) else _parse_bed(peaks)
        fasta = genome if isinstance(genome, pyfaidx.Fasta) else pyfaidx.Fasta(str(genome))

        if raw_intervals:
            # §5 "peak centering quality": fixed windows assume the center means
            # something; broad input peaks make that (and §2.7's positional
            # statistics) unreliable.
            median_width = float(np.median([iv.end - iv.start for iv in raw_intervals]))
            if median_width > 2 * size:
                warnings.warn(
                    f"SequenceStore.from_bed: median input peak width is {median_width:.0f} bp, more than twice "
                    f"size={size}; fixed-width centering (and central-enrichment statistics) may be unreliable "
                    "for broad or poorly-summited peaks -- consider summit-centered peaks or a larger size.",
                    stacklevel=2,
                )

        # Resolve every window first (exclusions in input order), then read
        # the genome in sorted, merged blocks: windows on the same chromosome
        # closer than `_BLOCK_GAP` share one pyfaidx slice. One seek per
        # block instead of per peak (thousands -> hundreds on a cold shared
        # filesystem, where per-peak seeks cost ~10 ms), without reading
        # whole chromosomes. Kept order is the input order either way.
        windows: list[tuple[int, str, int, int]] = []  # (input index, chrom, start, end)
        kept_intervals: list[Interval] = []
        excluded_idx: list[tuple[int, Exclusion]] = []  # (input index, exclusion): reported in input order
        chrom_lens: dict[str, int] = {}
        for i, iv in enumerate(raw_intervals):
            if iv.chrom not in fasta:
                excluded_idx.append((i, Exclusion(iv, "unknown_chrom")))
                continue
            if iv.chrom not in chrom_lens:
                chrom_lens[iv.chrom] = len(fasta[iv.chrom])
            center = (iv.start + iv.end) // 2
            new_start = center - size // 2
            new_end = new_start + size
            if new_start < 0 or new_end > chrom_lens[iv.chrom]:
                excluded_idx.append((i, Exclusion(iv, "chrom_edge")))
                continue
            windows.append((i, iv.chrom, new_start, new_end))

        codes_by_input: dict[int, np.ndarray] = {}
        order = sorted(range(len(windows)), key=lambda j: (windows[j][1], windows[j][2]))
        j = 0
        while j < len(order):
            _, chrom, block_start, block_end = windows[order[j]]
            k = j + 1
            while k < len(order):
                _, c, s, e = windows[order[k]]
                if c != chrom or s - block_end > _BLOCK_GAP:
                    break
                block_end = max(block_end, e)
                k += 1
            block = encode(fasta[chrom][block_start:block_end].seq)
            for idx in order[j:k]:
                i, _, s, e = windows[idx]
                codes_by_input[i] = block[s - block_start : e - block_start].copy()
            j = k

        kept_codes: list[np.ndarray] = []
        for i, chrom, new_start, new_end in windows:
            iv = raw_intervals[i]
            codes = codes_by_input[i]
            n_frac = (codes == _N_CODE).mean()
            if n_frac > max_n_frac:
                excluded_idx.append((i, Exclusion(iv, "max_n_frac")))
                continue

            kept_codes.append(codes)
            kept_intervals.append(Interval(iv.chrom, new_start, new_end, iv.name))

        excluded = [exc for _, exc in sorted(excluded_idx, key=lambda t: t[0])]
        if excluded:
            by_reason: dict[str, int] = {}
            for exc in excluded:
                by_reason[exc.reason] = by_reason.get(exc.reason, 0) + 1
            warnings.warn(
                f"SequenceStore.from_bed: dropped {len(excluded)}/{len(raw_intervals)} "
                f"peaks ({by_reason}). See the returned store's `.excluded` for detail.",
                stacklevel=2,
            )
        if not kept_codes:
            raise ValueError("SequenceStore.from_bed: no peaks survived extraction")

        codes_arr = np.stack(kept_codes)
        codes_t = torch.from_numpy(codes_arr).to(device=device, dtype=torch.uint8)
        mask_t = codes_t != _N_CODE
        return cls(codes_t, mask_t, kept_intervals, excluded, device=device)

    @classmethod
    def from_fasta(cls, path, device: str = "cpu", max_n_frac: float = 0.2) -> "SequenceStore":
        """Sequences from a FASTA of already-equal-length records.

        Unlike `from_bed` there is no window to re-center: every record is
        used as-is (and must be the same length as the first record), only
        the `max_n_frac` filter applies.
        """
        import pyfaidx

        fasta = pyfaidx.Fasta(str(path))
        kept_codes: list[np.ndarray] = []
        kept_intervals: list[Interval] = []
        excluded: list[Exclusion] = []
        length: int | None = None

        for name in fasta.keys():
            seq = str(fasta[name])
            if length is None:
                length = len(seq)
            elif len(seq) != length:
                raise ValueError(
                    f"from_fasta requires equal-length records; '{name}' has length "
                    f"{len(seq)}, expected {length}"
                )
            codes = encode(seq)
            iv = Interval(name, 0, length, name)
            n_frac = (codes == _N_CODE).mean()
            if n_frac > max_n_frac:
                excluded.append(Exclusion(iv, "max_n_frac"))
                continue
            kept_codes.append(codes)
            kept_intervals.append(iv)

        if excluded:
            warnings.warn(
                f"SequenceStore.from_fasta: dropped {len(excluded)}/{len(kept_codes) + len(excluded)} "
                "records for exceeding max_n_frac.",
                stacklevel=2,
            )
        if not kept_codes:
            raise ValueError("SequenceStore.from_fasta: no records survived filtering")

        codes_arr = np.stack(kept_codes)
        codes_t = torch.from_numpy(codes_arr).to(device=device, dtype=torch.uint8)
        mask_t = codes_t != _N_CODE
        return cls(codes_t, mask_t, kept_intervals, excluded, device=device)

    @classmethod
    def from_sequences(cls, seqs: list[str], device: str = "cpu") -> "SequenceStore":
        """Build directly from a list of equal-length ACGTN strings, no FASTA/BED.

        For in-memory sequences that didn't come from a genome+peaks pair --
        e.g. `control.py`'s dinucleotide-shuffle and GC-matched pools, or
        tests. `intervals` get a synthetic "control"/`seq_i` name since there
        are no real genomic coordinates to record.
        """
        if not seqs:
            raise ValueError("from_sequences requires at least one sequence")
        length = len(seqs[0])
        if any(len(s) != length for s in seqs):
            raise ValueError("from_sequences requires all sequences to be the same length")
        codes_arr = np.stack([encode(s) for s in seqs])
        codes_t = torch.from_numpy(codes_arr).to(device=device, dtype=torch.uint8)
        mask_t = codes_t != _N_CODE
        intervals = [Interval("control", 0, length, f"seq_{i}") for i in range(len(seqs))]
        return cls(codes_t, mask_t, intervals, excluded=[], device=device)

    def subset(self, idx) -> "SequenceStore":
        """A new store over a subset of rows, keeping the fitted background
        (indexed the same way) if one exists rather than dropping it --
        useful for e.g. `MotifDiscovery.enrichment`'s `idx=` and
        `control.Explicit`'s index-array form, where re-fitting on the
        subset would silently change what "background" means.
        """
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.codes.device)
        new = SequenceStore(
            self.codes[idx_t].clone(),
            self.mask[idx_t].clone(),
            [self.intervals[i] for i in idx_t.tolist()],
            excluded=[],
            device=self.device,
        )
        if self.bg_cumsum is not None:
            new.bg_cumsum = self.bg_cumsum[idx_t].clone()
            new.bg_table = self.bg_table
            new.bg_marginals = self.bg_marginals
            new.bg_tables = self.bg_tables
            new.bg_model = self.bg_model
            new.bg_order = self.bg_order
        return new

    def to_strings(self) -> list[str]:
        """Decode every row back to an ACGTN string (masked/erased positions read as `N`)."""
        lut = np.array(list(_ALPHABET + "N"))
        codes = self.codes.cpu().numpy()
        return ["".join(lut[row]) for row in codes]

    def to_fasta(self, path) -> None:
        """Write the store as FASTA, one record per row named after its
        interval (`chrom:start-end` plus the BED name) -- the round trip
        needed to hand the same sequences to an external tool (STREME/FIMO
        for the L1/L6 comparisons).
        """
        with open(path, "w") as fh:
            for iv, seq in zip(self.intervals, self.to_strings()):
                fh.write(f">{iv.chrom}:{iv.start}-{iv.end}" + (f" {iv.name}" if iv.name else "") + f"\n{seq}\n")

    def erase(self, rows, starts, width: int) -> int:
        """Mask out the windows `[starts[i], starts[i]+width)` of `rows[i]`, in
        place: the bases become `N` (code 4, mask False), so nothing built on
        this store afterwards -- k-mer counts, scan windows, background fits
        -- can see them. This is STREME's "erasing" of a found motif's sites
        before the next round, so the next motif isn't the same one again.

        Invalidates the fitted background (`bg_cumsum` is set back to `None`;
        call `fit_background` again) and the one-hot cache. Returns the
        number of windows erased.
        """
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.codes.device).reshape(-1)
        starts_t = torch.as_tensor(starts, dtype=torch.long, device=self.codes.device).reshape(-1)
        if rows_t.numel() != starts_t.numel():
            raise ValueError("erase: rows and starts must have the same length")
        if rows_t.numel() == 0:
            return 0
        if width < 1 or int(starts_t.min()) < 0 or int(starts_t.max()) + width > self.length:
            raise ValueError("erase: window runs outside the sequence")
        offsets = torch.arange(width, device=self.codes.device)
        cols = starts_t.unsqueeze(1) + offsets.unsqueeze(0)  # (n, width)
        rows_b = rows_t.unsqueeze(1).expand_as(cols)
        self.codes[rows_b, cols] = _N_CODE
        self.mask[rows_b, cols] = False
        self.bg_cumsum = None
        self._onehot_cache.clear()
        return int(rows_t.numel())

    # ------------------------------------------------------------------
    # Derived representations
    # ------------------------------------------------------------------

    def onehot(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """(N, 4, L) one-hot encoding, cached per dtype. N positions are all-zero."""
        if dtype not in self._onehot_cache:
            idx = self.codes.long().clamp(max=3)
            oh = torch.zeros(self.n_seqs, self.length, 4, dtype=dtype, device=self.device)
            oh.scatter_(2, idx.unsqueeze(-1), 1)
            oh = oh * self.mask.unsqueeze(-1).to(dtype)
            self._onehot_cache[dtype] = oh.permute(0, 2, 1).contiguous()
        return self._onehot_cache[dtype]

    # ------------------------------------------------------------------
    # Background model (§2.3)
    # ------------------------------------------------------------------

    def background_model(self, order: int = 2, pseudocount: float = 1.0) -> "BackgroundModel":
        """Estimate an order-`order` Markov background from *this* store's
        sequences and return it, without scoring anything.

        Estimation and application are separate because the round loop
        applies one model to four or five stores per round (training and
        hold-out primary/control, plus the full primary set it reports
        from). Re-estimating per store made every round count the same
        control set that many times over -- and since the graded fallback
        needs a table per order 0..k, that redundancy multiplied. Fit once,
        `apply_background` many.
        """
        codes = self.codes.to(torch.long)
        counts0 = torch.bincount(codes[self.mask], minlength=4).float() + pseudocount
        log_freq0 = torch.log(counts0 / counts0.sum())
        tables = [log_freq0]

        if order > 0:
            if codes.shape[1] <= order:
                raise ValueError(
                    f"bg_order={order} requires sequences longer than {order}; "
                    f"got background source length {codes.shape[1]}"
                )
            # `clamp(max=3)` reads an N (code 4) as a T; the counting side is
            # unaffected because `valid` drops every window touching one.
            clamped = codes.clamp(max=3)
            for j in range(1, order + 1):
                powers_j = (4 ** torch.arange(j - 1, -1, -1, device=codes.device)).long()
                win = clamped.unfold(1, j + 1, 1)  # (N, L-j, j+1)
                valid = self.mask.unfold(1, j + 1, 1).all(dim=2)
                ctx = (win[:, :, :j] * powers_j).sum(dim=2)
                base = win[:, :, j]
                combined = ctx[valid] * 4 + base[valid]
                counts = torch.bincount(combined, minlength=4**j * 4).float().reshape(4**j, 4) + pseudocount
                tables.append(torch.log(counts / counts.sum(dim=1, keepdim=True)))  # (4**j, 4)

        return BackgroundModel(
            order=order,
            tables=tables,
            pseudocount=pseudocount,
            n_seqs=self.n_seqs,
            length=self.length,
        )

    def apply_background(self, model: "BackgroundModel") -> None:
        """Score this store's own codes/mask under an already-estimated
        `model` and set `bg_cumsum` (plus `bg_order`/`bg_table`/
        `bg_tables`/`bg_marginals`/`bg_model`).

        A position with fewer than `model.order` valid bases of preceding
        context -- near the start of a sequence, or after an N or an erased
        base -- backs off to the highest order its context does support,
        and to the order-0 marginal only when the base immediately before
        it is invalid. Any window touching an N is masked out at scan time
        regardless of what value ends up here, so N positions are given a
        value of 0 rather than something meaningful -- but the valid
        positions *after* an N are scored for real, which is why their
        context cannot be silently read as T.
        """
        codes = self.codes.to(torch.long)
        length = codes.shape[1]
        k = model.order
        if k > 0 and length <= k:
            raise ValueError(f"bg_order={k} requires sequences longer than {k}; got length {length}")
        tables = [t.to(codes.device) for t in model.tables]  # a model may be fitted on another device

        clamped = codes.clamp(max=3)
        bg_ll = tables[0][clamped]
        # order 0 everywhere, then overwrite each position with the highest
        # order whose context is entirely valid. `ok` is nested in j (an
        # order-j context contains the order-(j-1) one), so the last write to
        # a position is the highest order it qualifies for.
        for j in range(1, k + 1):
            powers_j = (4 ** torch.arange(j - 1, -1, -1, device=codes.device)).long()
            win = clamped.unfold(1, j, 1)[:, : length - j, :]  # bases i-j..i-1, for i = j..L-1
            ctx = (win * powers_j).sum(dim=2)
            base = clamped[:, j:]
            ok = self.mask.unfold(1, j, 1).all(dim=2)[:, : length - j].to(codes.device)
            bg_ll[:, j:] = torch.where(ok, tables[j][ctx, base], bg_ll[:, j:])

        bg_ll = torch.where(self.mask, bg_ll.to(self.mask.device), torch.zeros_like(bg_ll))
        zeros = torch.zeros(self.n_seqs, 1, dtype=bg_ll.dtype, device=bg_ll.device)
        self.bg_cumsum = torch.cat([zeros, torch.cumsum(bg_ll, dim=1)], dim=1)  # (N, L+1)
        self.bg_order = k
        self.bg_model = model
        self.bg_tables = tables
        self.bg_table = tables[-1]  # the top-order table (the marginals, at order 0), as before
        self.bg_marginals = tables[0]

    def fit_background(self, order: int = 2, source: "SequenceStore | None" = None, pseudocount: float = 1.0):
        """Fit an order-`order` Markov background and set `bg_cumsum`.

        `source` defaults to `self` (fit on the sequences being scored);
        DESIGNDOC.md §2.3 recommends fitting on the *control* set once one
        exists (Phase 2), by passing that store's SequenceStore here instead
        -- the tables are estimated from `source`'s sequences, then applied
        to `self`'s own codes/mask to produce `self.bg_cumsum`. `source`
        need not have the same number of sequences or the same length as
        `self`.

        This is `background_model` followed by `apply_background`, kept as
        the one-liner for the common single-store case. Call the two halves
        directly when one model is applied to several stores -- see
        `background_model` for why that matters.
        """
        model = (source if source is not None else self).background_model(order, pseudocount)
        self.apply_background(model)

    def background_window_sum(self, start: int, width: int) -> torch.Tensor:
        """Sum of background log-likelihood over `[start, start+width)`, all sequences.

        Requires `fit_background` to have been called first.
        """
        if self.bg_cumsum is None:
            raise RuntimeError("call fit_background() before background_window_sum()")
        return self.bg_cumsum[:, start + width] - self.bg_cumsum[:, start]

    def background_window_sums(self, width: int) -> torch.Tensor:
        """Background log-likelihood sum for every window of `width`, all starts at once.

        Shape (N, L-width+1). This is the vectorized form the scanner uses:
        one subtraction of two cumsum slices covers every window position at
        once, regardless of background order (DESIGNDOC.md §2.3).
        """
        if self.bg_cumsum is None:
            raise RuntimeError("call fit_background() before background_window_sums()")
        return window_sums_from_cumsum(self.bg_cumsum, width)


def window_sums_from_cumsum(cumsum: torch.Tensor, width: int) -> torch.Tensor:
    """Shared cumsum-difference trick: (N, L+1) -> (N, L-width+1) window sums."""
    return cumsum[:, width:] - cumsum[:, : cumsum.shape[1] - width]
