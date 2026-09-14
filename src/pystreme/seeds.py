"""Seed finding: k-mer counting and ranking.

See DESIGNDOC.md §2.1 (direct counting replaces the suffix tree for this job)
and §5 (memory: 4^w count arrays, sort-based fallback above w=12). Phase 3.

Words are counted pooled with their reverse complement -- canonicalized to
whichever of {word, revcomp(word)} sorts first as a packed integer -- since
the scanner searches both strands (§2.4): a word and its RC are the same
candidate motif, not two different ones splitting the same signal.

That holds only while the scan really is double-stranded. Under
`revcomp=False` the two orientations are *not* the same candidate, and
emitting the lexicographically smaller one hands refinement a seed the
forward-only scanner can never match, so words are then counted in the
orientation they were observed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from .sequence_store import SequenceStore
from .statistics import binom_logsf, hypergeom_logsf

_ALPHABET = "ACGT"
_DENSE_COUNT_MAX_W = 12  # 4**12 ~ 16.7M entries; the design doc's own cutoff


@dataclass
class Seed:
    width: int
    word: str
    primary_count: int
    control_count: int
    log_pvalue: float


def _kmer_ids(codes: torch.Tensor, mask: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every valid length-`w` window's packed base-4 id and validity, each
    shape (N, L-w+1). Shared with `_revcomp_id`/`_digits_to_word` below.
    """
    n_seqs, length = codes.shape
    if length < w:
        raise ValueError(f"width {w} is longer than the sequence length {length}")
    clamped = codes.long().clamp(max=3)
    windows = clamped.unfold(1, w, 1)  # (N, L-w+1, w)
    valid = mask.unfold(1, w, 1).all(dim=2)  # (N, L-w+1)
    powers = (4 ** torch.arange(w - 1, -1, -1, device=codes.device)).long()
    ids = (windows * powers).sum(dim=2)
    return ids, valid


def _digits(ids: torch.Tensor, w: int) -> torch.Tensor:
    """Packed id(s) -> base-4 digits, shape (..., w); digit 0 is the first (most significant) base."""
    powers = 4 ** torch.arange(w - 1, -1, -1, device=ids.device)
    return (ids.unsqueeze(-1) // powers) % 4


def _encode_digits(digits: torch.Tensor, w: int) -> torch.Tensor:
    powers = 4 ** torch.arange(w - 1, -1, -1, device=digits.device)
    return (digits * powers).sum(dim=-1)


def _revcomp_id(ids: torch.Tensor, w: int) -> torch.Tensor:
    """Packed id of the reverse complement of the word(s) `ids` encodes:
    complement each base (A<->T, C<->G is `3 - digit`) *and* reverse their
    order (RC flips 5'->3' direction, not just base identity)."""
    digits = _digits(ids, w)
    return _encode_digits((3 - digits).flip(dims=[-1]), w)


def _id_to_word(id_: int, w: int) -> str:
    digits = _digits(torch.tensor([id_]), w)[0].tolist()
    return "".join(_ALPHABET[d] for d in digits)


def top_seeds(
    primary_store: SequenceStore,
    control_store: SequenceStore,
    widths,
    n_per_width: int = 4,
    n_exact: int = 2000,
    revcomp: bool = True,
) -> dict[int, list[Seed]]:
    """Rank candidate seed words per width by primary-vs-control enrichment.

    One vectorized counting pass per width (§2.1): direct dense `bincount`
    into a `4**w`-sized array when that fits comfortably (`w <= 12`, the
    design doc's own cutoff -- `4**12` is ~16.7M entries), else a sort/unique
    -based count over just the words actually observed (packed ids fit an
    int64 up to w=31, far past the default 6-15 width range, so nothing
    changes about *how* words are represented -- only how they're counted).

    `revcomp` must match the scan the seeds are refined under: with
    `False`, words are counted in their observed orientation rather than
    pooled with their reverse complements (see the module docstring).

    Enrichment is a single vectorized `statistics.hypergeom_logsf` call
    across every candidate word at once (the same one-sided "pool of primary+control
    occurrences, how many landed in primary" test `statistics.py` uses for
    score thresholds, applied here to raw word-occurrence counts instead of
    a scan-score cutoff).
    """
    results: dict[int, list[Seed]] = {}
    for w in widths:
        p_ids, p_valid = _kmer_ids(primary_store.codes, primary_store.mask, w)
        c_ids, c_valid = _kmer_ids(control_store.codes, control_store.mask, w)
        if revcomp:
            # a word and its RC are one candidate: pool them under whichever
            # packs to the smaller integer
            p_canon = torch.minimum(p_ids, _revcomp_id(p_ids, w))[p_valid]
            c_canon = torch.minimum(c_ids, _revcomp_id(c_ids, w))[c_valid]
        else:
            # forward-only: keep the orientation actually observed, or the
            # seed handed to refinement can be the one strand the scanner
            # will never look at
            p_canon = p_ids[p_valid]
            c_canon = c_ids[c_valid]

        # Only words seen in primary can be seeds (and they're already
        # canonical ids by construction), so neither path ever enumerates
        # the full 4**w space: the dense path bincounts into it (134 MB of
        # int64 at w=12, the design doc's cutoff) and keeps the nonzero
        # entries; the sort path never allocates it at all.
        n_ids = 4**w
        if n_ids <= 4**_DENSE_COUNT_MAX_W:
            p_counts_all = torch.bincount(p_canon, minlength=n_ids)
            c_counts_all = torch.bincount(c_canon, minlength=n_ids)
            candidate_ids = torch.nonzero(p_counts_all > 0).squeeze(1)
            p_counts = p_counts_all[candidate_ids]
            c_counts = c_counts_all[candidate_ids]
        else:
            candidate_ids, inverse = torch.unique(torch.cat([p_canon, c_canon]), return_inverse=True)
            n_p = p_canon.numel()
            p_counts = torch.bincount(inverse[:n_p], minlength=candidate_ids.numel())
            c_counts = torch.bincount(inverse[n_p:], minlength=candidate_ids.numel())
            seen = p_counts > 0
            candidate_ids, p_counts, c_counts = candidate_ids[seen], p_counts[seen], c_counts[seen]

        if candidate_ids.numel() == 0:
            results[w] = []
            continue

        pool = int(p_valid.sum()) + int(c_valid.sum())
        successes = (p_counts + c_counts).cpu().numpy()
        draws = int(p_valid.sum())
        p_np = p_counts.cpu().numpy()

        # The exact hypergeometric tail costs O(draws) per word (scipy's C
        # `sf` sums pmf terms), which with ~1e5-1e6 candidate words against
        # a pool of ~1e6 windows was >90% of fit()'s runtime. Pre-rank every
        # word with the O(1) binomial tail (incomplete beta) at the pooled
        # rate -- same ordering to a very good approximation -- and spend
        # the exact test only on the `n_exact` best.
        cand = np.arange(p_np.size)
        if p_np.size > n_exact:
            approx = binom_logsf(p_np - 1, successes, draws / pool)
            cand = np.argpartition(approx, n_exact)[:n_exact]
        log_pvals = hypergeom_logsf(p_np[cand] - 1, pool, successes[cand], draws)

        # ties on p-value (shifted copies of one word, exact same counts) are
        # broken by word id, so the ranking is the same whichever candidate
        # subset the pre-filter handed in -- argsort alone is not stable
        order = cand[np.lexsort((candidate_ids.cpu().numpy()[cand], log_pvals))[:n_per_width]]
        log_pvals_by_index = dict(zip(cand.tolist(), log_pvals.tolist()))
        seeds = [
            Seed(
                width=w,
                word=_id_to_word(int(candidate_ids[i]), w),
                primary_count=int(p_counts[i]),
                control_count=int(c_counts[i]),
                log_pvalue=log_pvals_by_index[int(i)],
            )
            for i in order
        ]
        results[w] = seeds
    return results
