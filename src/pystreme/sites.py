"""Motif sites with moodspy: every hit above a p-value, not one per peak.

`fit` keeps its own batched scorer (`pystreme.scanner.scan`): refinement,
thresholds, hold-out and erasing need every peak's best window scored
against the order-k Markov background, hundreds of times per round, which
is what that scanner is built for. Reporting sites is a different job:
which placements in a peak are good matches to a finished motif. That is
done here with moodspy (MOODS: exact p-value thresholds, lookahead
filtration), so a peak can carry several sites, or none, and the cutoff
means the same thing for every motif.

Scores are log2 odds (bits) against an order-0 background, the MOODS/FIMO
convention. They are not comparable with `Motif.threshold` or
`Motif.sites.scores`, which are natural-log ratios against the order-k
background `fit` scored with.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .scanner import log_probabilities
from .sequence_store import _N_CODE, SequenceStore, as_row_index

_ALPHABET = "ACGT"
DEFAULT_P_CUTOFF = 1e-4  # FIMO's default --thresh


def _moodspy():
    try:
        import moodspy
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "motif sites are called with moodspy: "
            "`pip install \"moodspy @ git+https://github.com/vmindel/moodspy\"` "
            "(or install pystreme with the [sites] extra)"
        ) from exc
    return moodspy


def _named_pwms(motifs) -> list[tuple[str, np.ndarray]]:
    """Accepts a (4, w) array, anything with `.pwm`/`.name` (a fitted
    `Motif`, a `MemeMotif`), a {name: pwm} mapping, or a list of these."""
    if isinstance(motifs, Mapping):
        return [(str(k), np.asarray(v, dtype=np.float64)) for k, v in motifs.items()]
    if (isinstance(motifs, np.ndarray) and motifs.ndim == 2) or hasattr(motifs, "pwm"):
        motifs = [motifs]
    out = []
    for i, m in enumerate(motifs):
        if hasattr(m, "pwm"):
            out.append((str(m.name), np.asarray(m.pwm, dtype=np.float64)))
        else:
            out.append((f"motif_{i + 1}", np.asarray(m, dtype=np.float64)))
    for name, pwm in out:
        if pwm.ndim != 2 or pwm.shape[0] != 4:
            raise ValueError(f"motif {name!r}: expected a (4, w) probability matrix, got shape {pwm.shape}")
    return out


def store_background(store: SequenceStore) -> np.ndarray:
    """Order-0 A/C/G/T frequencies for scoring sites on `store`: the
    marginals of its fitted background when it has one, else the base
    composition of all its unmasked bases. Always the whole store's, so a
    scan of a subset scores exactly as the same rows do in a full scan."""
    if store.bg_marginals is not None:
        bg = store.bg_marginals.exp().double().cpu().numpy()
    else:
        codes = store.codes[store.mask].long().cpu().numpy()
        bg = np.bincount(codes, minlength=4)[:4].astype(np.float64) + 1.0
    return bg / bg.sum()


def find_sites(
    motifs,
    store: SequenceStore,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    background=None,
    idx=None,
    n_jobs: int | None = None,
):
    """Every placement of each motif, on both strands, whose score reaches
    the motif's p-value cutoff (moodspy's exact DP threshold).

    `motifs` is a probability PWM (4, w), a fitted `Motif`, a `MemeMotif`,
    a {name: pwm} dict, or a list of these. `background` is 4 A/C/G/T
    frequencies; by default the store's order-0 marginals (see
    `store_background`). `idx` restricts the scan to those rows of
    `store`. Bases that are N or erased in the store never fall inside a
    site.

    Returns a DataFrame, one row per site: `motif`, `row` (index into
    `store`), `peak` (interval name), `chrom`, `start`, `end` (the site,
    0-based half-open, forward-strand genome coordinates), `strand`,
    `score` (log2 odds, bits) and `offset` (site centre minus window
    centre, in bases, as in `Motif.positions`). Sorted by motif, row,
    start.
    """
    mp = _moodspy()
    import pandas as pd

    named = _named_pwms(motifs)
    bg = store_background(store) if background is None else np.asarray(background, dtype=np.float64)
    if bg.shape != (4,) or (bg <= 0).any():
        raise ValueError(f"background must be 4 positive A/C/G/T frequencies, got {bg}")
    bg = bg / bg.sum()

    rows = np.arange(store.n_seqs) if idx is None else as_row_index(idx, store.n_seqs).cpu().numpy()
    codes = store.codes.cpu().numpy()[rows].copy()
    codes[~store.mask.cpu().numpy()[rows]] = _N_CODE
    lut = np.frombuffer((_ALPHABET + "N").encode("ascii"), dtype=np.uint8)
    seqs = [row.tobytes().decode("ascii") for row in lut[codes]]

    # log2(p / bg) of the same clipped matrix `fit` scores, passed to moodspy
    # as log-odds so it applies no smoothing of its own
    mp_motifs = [
        mp.Motif(log_probabilities(pwm) / np.log(2.0) - np.log2(bg)[:, None], name=f"m{k}", kind="log-odds", background=bg)
        for k, (_, pwm) in enumerate(named)
    ]
    hits = mp.Scanner(mp_motifs, background=bg, p_cutoff=p_cutoff, n_jobs=n_jobs).scan(seqs)

    names = np.array([n for n, _ in named], dtype=object)
    widths = np.array([pwm.shape[1] for _, pwm in named])
    k = hits["motif"].cat.codes.to_numpy()
    r = rows[hits["seq"].to_numpy()]
    ivs = store.intervals
    iv_start = np.array([ivs[i].start for i in r], dtype=np.int64)
    pos = hits["start"].to_numpy()
    length = store.length
    df = pd.DataFrame(
        {
            "motif": pd.Categorical(names[k], categories=list(dict.fromkeys(names))),
            "row": r,
            "peak": [ivs[i].name for i in r],
            "chrom": [ivs[i].chrom for i in r],
            "start": iv_start + pos,
            "end": iv_start + pos + widths[k],
            "strand": hits["strand"].astype(str).to_numpy(),
            "score": hits["score"].to_numpy(),
            "offset": pos + (widths[k] - 1) / 2 - (length - 1) / 2,
        }
    )
    order = np.lexsort((df["strand"].to_numpy() == "-", df["start"].to_numpy(), r, k))
    return df.iloc[order].reset_index(drop=True)


def write_sites_bed(sites, path, sort: bool = True) -> int:
    """`find_sites` output as BED6: `chrom start end motif|peak score
    strand`, score in bits. Sorted by chromosome and start unless
    `sort=False`. Returns the number of lines written."""
    df = sites.sort_values(["chrom", "start", "end", "motif", "peak"], kind="stable") if sort else sites
    with open(path, "w") as fh:
        for r in df.itertuples(index=False):
            fh.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.motif}|{r.peak}\t{r.score:.3f}\t{r.strand}\n")
    return len(df)
