"""PWM scanning: the batched-convolution scanner.

See DESIGNDOC.md §2.1 (best-site lookup as convolution), §2.3 (background
subtraction via cumsum, split from the motif term), §2.4 (reverse complement
as a second kernel), §3.1 (two scan kernels, benchmarked not assumed). Phase 1.

score(window at i) = sum_j log P_motif(x_{i+j} | j)   -- the PWM kernel term
                    - sum_j log P_bg(x_{i+j} | context) -- store.bg_cumsum

The two terms are independent (the background one doesn't depend on the
motif at all), so they're computed by different mechanisms and just
subtracted: a convolution/gather for the first, a cumsum difference for the
second.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .sequence_store import SequenceStore, as_row_index, window_sums_from_cumsum


@dataclass
class ScanResult:
    scores: torch.Tensor  # (n_motifs, n_seqs) fp32; -inf where a motif has no valid window
    positions: torch.Tensor  # (n_motifs, n_seqs) long, 0-based window-start offset
    strands: torch.Tensor  # (n_motifs, n_seqs) int8, +1 or -1


_TIE_EPS = 1e-4  # nats; window scores closer than this are a tie, broken by rule rather than by rounding


def _pick_best(fwd: torch.Tensor, rc: torch.Tensor | None):
    """Best window per (sequence, motif) from per-window scores (N, M, W).

    Returns `(score, position, strand)`, each (N, M). Near-ties are broken by
    rule: the forward strand unless the reverse strand wins by more than
    `_TIE_EPS`, then the leftmost window within `_TIE_EPS` of the best.
    Identical windows (a site planted next to a chance copy of itself, a
    palindrome's two strands) score the same in exact arithmetic, but the
    background cumsum and the conv/gather kernels round differently at
    different positions and on different CPUs, so an exact `max` lets
    last-bit noise choose -- and the choice moves the reported site, which
    `fit` then erases. The reported score is the chosen window's own.
    """
    use_rc = None
    if rc is None:
        combined = fwd
    else:
        use_rc = rc > fwd + _TIE_EPS
        combined = torch.where(use_rc, rc, fwd)
    top = combined.max(dim=2, keepdim=True).values
    # argmax returns the first maximal index (documented), so this is the
    # leftmost near-best window; rows with no valid window are all -inf,
    # every entry counts as "near", and position 0 / score -inf come out
    near = combined >= top - _TIE_EPS
    pos = near.to(torch.uint8).argmax(dim=2)
    score = torch.gather(combined, 2, pos.unsqueeze(2)).squeeze(2)
    strand = torch.ones_like(pos, dtype=torch.int8)
    if use_rc is not None:  # looked up at the chosen window only, not built for every window
        strand[torch.gather(use_rc, 2, pos.unsqueeze(2)).squeeze(2)] = -1
    return score, pos, strand


def _to_log_pwm(pwm: np.ndarray, eps: float = 1e-4) -> torch.Tensor:
    """Probability PWM (4, w) -> log-probability kernel, clipped to avoid log(0)."""
    pwm = np.clip(np.asarray(pwm, dtype=np.float64), eps, 1.0)
    pwm = pwm / pwm.sum(axis=0, keepdims=True)
    return torch.from_numpy(np.log(pwm)).to(torch.float32)


def _default_kernel(device: torch.device) -> str:
    # §3.1 "benchmarked, not assumed" -- scripts/bench_scan.py, recorded in
    # BENCHMARKS.md. CPU: conv (fp32 there, so no precision caveat) is ~7x
    # faster than gather per scan. CUDA: conv (fp16 one-hot) is ~2x faster
    # per scan at 40 PWMs, but fit() scans a handful of PWMs at a time and
    # its wall-clock was the same with either (9.3 s gather vs 10.2 s conv
    # on 2000 peaks); gather is fp32 end to end, so it is the CUDA default
    # and conv stays selectable with kernel="conv".
    return "gather" if device.type == "cuda" else "conv"


def _scan_conv(onehot: torch.Tensor, log_pwms: torch.Tensor) -> torch.Tensor:
    """onehot (N,4,L), log_pwms (M,4,w) -> (N,M,L-w+1). Cross-correlation == matched filter."""
    return F.conv1d(onehot.to(log_pwms.dtype), log_pwms)


def _scan_gather(codes: torch.Tensor, log_pwms: torch.Tensor) -> torch.Tensor:
    """codes (N,L) uint8/long, log_pwms (M,4,w) -> (N,M,L-w+1).

    score[n,m,i] = sum_j log_pwms[m, codes[n,i+j], j] -- one table lookup
    per offset j, as described in DESIGNDOC.md §3.1. Accumulated as
    (M, N, windows) so the window axis stays contiguous through the
    permute: the max over windows that follows reads it in order.
    """
    n_motifs, _, w = log_pwms.shape
    n_seqs, length = codes.shape
    n_win = length - w + 1
    x = codes.long().clamp(max=3)
    table = log_pwms.permute(2, 0, 1)  # (w, M, 4)

    out = torch.zeros(n_motifs, n_seqs, n_win, dtype=log_pwms.dtype, device=codes.device)
    for j in range(w):
        out += table[j][:, x[:, j : j + n_win]]
    return out.permute(1, 0, 2)  # (N, M, n_win)


def _run_kernel(kernel: str, store_slice, log_pwms: torch.Tensor, onehot: torch.Tensor | None = None) -> torch.Tensor:
    codes, mask = store_slice
    if kernel == "conv":
        dtype = torch.float16 if log_pwms.device.type == "cuda" else torch.float32
        if onehot is None:
            onehot = _onehot_from_codes(codes, mask, dtype=dtype)
        return _scan_conv(onehot, log_pwms.to(dtype)).float()
    elif kernel == "gather":
        return _scan_gather(codes, log_pwms)
    else:
        raise ValueError(f"unknown kernel {kernel!r}; expected 'conv', 'gather', or 'auto'")


def _onehot_from_codes(codes: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_seqs, length = codes.shape
    idx = codes.long().clamp(max=3)
    oh = torch.zeros(n_seqs, length, 4, dtype=dtype, device=codes.device)
    oh.scatter_(2, idx.unsqueeze(-1), 1)
    oh = oh * mask.unsqueeze(-1).to(dtype)
    return oh.permute(0, 2, 1).contiguous()  # (N, 4, L)


def valid_windows(mask: torch.Tensor, w: int) -> torch.Tensor:
    """(N, L) base mask -> (N, L-w+1): True where the whole window is real
    sequence. A cumulative count of invalid bases, so O(N L) whatever `w`
    (the `unfold(...).all()` it replaces cost O(N L w))."""
    bad = torch.cumsum((~mask).to(torch.int32), dim=1)
    bad = torch.cat([torch.zeros_like(bad[:, :1]), bad], dim=1)
    return (bad[:, w:] - bad[:, : bad.shape[1] - w]) == 0


def _masked_background(mask: torch.Tensor, bg_cumsum: torch.Tensor, w: int) -> torch.Tensor:
    """Background window sums with +inf where a window touches an N or an
    erased base, so `kernel - this` is -inf there with no separate mask pass."""
    bg_win = window_sums_from_cumsum(bg_cumsum, w)
    return torch.where(valid_windows(mask, w), bg_win, torch.full_like(bg_win, float("inf")))


def _store_masked_background(store: SequenceStore, w: int) -> torch.Tensor:
    """`_masked_background` for the whole store, cached on it per width.

    Refinement scans the same store with new PWMs every iteration, and this
    term depends only on the store. The cache is checked against the
    background tensor's identity and the in-place version counters of the
    codes and mask, so `erase`, `apply_background`, and a view whose owner
    was erased all invalidate it without anyone having to remember to.
    """
    key = (store.bg_cumsum, store.mask._version, store.codes._version)
    cache = store.__dict__.setdefault("_masked_bg_cache", {})
    hit = cache.get(w)
    if hit is not None and hit[0] is key[0] and hit[1:] == key[1:]:
        return hit[3]
    value = _masked_background(store.mask, store.bg_cumsum, w)
    cache[w] = (*key, value)
    return value


def scan(
    pwms: list[np.ndarray],
    store: SequenceStore,
    idx: torch.Tensor | np.ndarray | None = None,
    revcomp: bool = True,
    kernel: str = "auto",
) -> ScanResult:
    """Batched PWM scan: best score/position/strand per (motif, sequence).

    `pwms` is a list of (4, w) probability matrices (as produced by
    `meme_io.read_meme`'s `MemeMotif.pwm`); motifs may have different widths.
    `store` must already have a background fit (`store.fit_background(...)`)
    -- there's no implicit default here, since silently picking one on your
    behalf is exactly the kind of surprise that makes a scan hard to trust.
    `idx` restricts the scan to a subset of sequences (row indices into
    `store`); `kernel` is `"conv"`, `"gather"`, or `"auto"` (device-based
    default, see `_default_kernel`).
    """
    if store.bg_cumsum is None:
        raise RuntimeError(
            "store has no background model; call store.fit_background(order=...) before scan()"
        )

    if idx is not None:
        idx = as_row_index(idx, store.n_seqs, device=store.codes.device)
    codes = store.codes if idx is None else store.codes[idx]
    mask = store.mask if idx is None else store.mask[idx]
    bg_cumsum = store.bg_cumsum if idx is None else store.bg_cumsum[idx]
    device = store.codes.device
    n_seqs = codes.shape[0]
    length = codes.shape[1]

    n_motifs = len(pwms)
    widths = [np.asarray(p).shape[1] for p in pwms]
    chosen_kernel = kernel if kernel != "auto" else _default_kernel(device)
    # the conv kernel's one-hot is a property of the store, not of the scan:
    # reuse the store's cache (invalidated by erase()) rather than rebuilding
    # (N, 4, L) on every refinement iteration
    onehot = None
    if chosen_kernel == "conv" and idx is None:
        onehot = store.onehot(torch.float16 if device.type == "cuda" else torch.float32)

    best_score = torch.full((n_motifs, n_seqs), float("-inf"), dtype=torch.float32, device=device)
    best_pos = torch.zeros((n_motifs, n_seqs), dtype=torch.long, device=device)
    best_strand = torch.ones((n_motifs, n_seqs), dtype=torch.int8, device=device)

    by_width: dict[int, list[int]] = defaultdict(list)
    for m, w in enumerate(widths):
        by_width[w].append(m)

    for w, m_indices in by_width.items():
        if w > length:
            continue  # motif wider than the sequence: no valid window anywhere, stays -inf

        log_pwms = torch.stack([_to_log_pwm(pwms[m]) for m in m_indices]).to(device)  # (Mg,4,w)
        n_group = len(m_indices)
        # +inf on invalid windows, so the subtraction below leaves -inf there
        bg = _store_masked_background(store, w) if idx is None else _masked_background(mask, bg_cumsum, w)

        # §2.4: the reverse complement is a second set of kernels in the same
        # call, not a second pass over the sequences
        kernels = torch.cat([log_pwms, torch.flip(log_pwms, dims=[1, 2])]) if revcomp else log_pwms
        raw = _run_kernel(chosen_kernel, (codes, mask), kernels, onehot) - bg.unsqueeze(1)
        fwd = raw[:, :n_group]
        rc = raw[:, n_group:] if revcomp else None

        max_score, max_pos, max_strand = _pick_best(fwd, rc)  # each (N, Mg)

        for local_i, m in enumerate(m_indices):
            best_score[m] = max_score[:, local_i]
            best_pos[m] = max_pos[:, local_i]
            best_strand[m] = max_strand[:, local_i]

    return ScanResult(scores=best_score, positions=best_pos, strands=best_strand)
