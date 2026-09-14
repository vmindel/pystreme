"""Results as tables: motif summaries, per-peak site tables, and database
annotation of discovered motifs. DESIGNDOC.md Phase 5 ("summary DataFrame
output").

pandas is imported lazily so the core package (scan/fit) has no dependency
on it; these functions raise a clear error if it is missing.
"""

from __future__ import annotations

import numpy as np

from .meme_io import MemeMotif, read_meme


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("pystreme.results needs pandas (`pip install pandas`)") from exc
    return pd


def sort_by_pvalue(motifs):
    """The motifs most significant first: by `holdout_logp`, falling back
    to the training `log_pvalue` for motifs without a hold-out. `fit()`
    returns discovery order (one motif per round), which is only roughly
    monotone in p -- each round is the best of what is *left* after
    erasing the previous motifs' sites, and the hold-out re-test adds
    noise -- so sort explicitly when the order in a table or figure should
    mean significance. Names keep their round prefix. Stable on ties."""
    if not isinstance(motifs, (list, tuple)):
        motifs = [motifs]

    def key(m):
        return m.holdout_logp if m.holdout_logp is not None else m.log_pvalue

    return sorted(motifs, key=key)


def summary(motifs, order: str = "round"):
    """One row per motif: the numbers you would put in a table or a
    figure legend. `holdout_logp` is the significance to quote (see
    `discovery.Motif`); `train_logp` is the selected, optimistic one.
    Works on `fit()` output, a single `discover()` result, or any mix.
    `order="pvalue"` sorts the rows most significant first (see
    `sort_by_pvalue`); the default keeps discovery order.
    """
    pd = _pandas()
    if not isinstance(motifs, (list, tuple)):
        motifs = [motifs]
    if order == "pvalue":
        motifs = sort_by_pvalue(motifs)
    elif order != "round":
        raise ValueError(f"summary: order must be 'pvalue' or 'round', got {order!r}")
    rows = []
    for m in motifs:
        rows.append(
            {
                "name": m.name,
                "round": m.round,
                "consensus": m.consensus,
                "width": m.width,
                "n_sites": m.n_sites,
                "train_logp": m.log_pvalue,
                "holdout_logp": m.holdout_logp,
                "central_logp": m.central_logp,
                "central_half_width": None if m.central is None else m.central.half_width,
                "threshold": m.threshold,
                "seed": m.seed,
            }
        )
    return pd.DataFrame(rows)


def sites_frame(motif):
    """One row per primary sequence the motif was scored on: the interval
    (as extracted, i.e. the fixed-width window), the best site's score,
    genomic start, strand and offset from the window center, and whether
    it clears the motif's threshold (`passing`; `passing.sum()` is
    `motif.n_sites`). Needs the intervals `fit`/`discover` attach to the
    motif (`Motif.intervals`).
    """
    pd = _pandas()
    if motif.intervals is None:
        raise ValueError("sites_frame: this Motif carries no intervals (built outside fit/discover)")
    scores = motif.sites.scores.cpu().numpy()
    positions = motif.sites.positions.cpu().numpy()
    strands = motif.sites.strands.cpu().numpy()
    ivs = motif.intervals
    if len(ivs) != scores.shape[0]:
        raise ValueError("sites_frame: intervals and sites disagree in length")
    lengths = np.array([iv.end - iv.start for iv in ivs])
    offsets = positions + (motif.width - 1) / 2 - (lengths - 1) / 2
    finite = np.isfinite(scores)
    return pd.DataFrame(
        {
            "chrom": [iv.chrom for iv in ivs],
            "start": [iv.start for iv in ivs],
            "end": [iv.end for iv in ivs],
            "name": [iv.name for iv in ivs],
            "score": scores,
            "site_start": np.where(finite, np.array([iv.start for iv in ivs]) + positions, -1),
            "site_end": np.where(finite, np.array([iv.start for iv in ivs]) + positions + motif.width, -1),
            "strand": np.where(strands < 0, "-", "+"),
            "offset": np.where(finite, offsets, np.nan),
            "passing": finite & (scores >= motif.threshold),
        }
    )


def sites_bed(motifs, path, passing_only: bool = True, sort: bool = True) -> int:
    """Write the motifs' sites as a BED6 file: `chrom start end name score
    strand`, one line per (motif, peak) pair -- the best site of that motif
    in that peak, its genomic coordinates (0-based, half-open, on the
    forward strand), `name` = `<motif name>|<peak name>`, `score` = the
    log-odds score in nats, `strand` = which strand the site matched.
    Only sites at or above each motif's threshold (`passing`) by default;
    `passing_only=False` writes every peak's best window. Returns the
    number of lines written. Sorted by chromosome and start unless
    `sort=False` (then motif order, peak order).
    """
    if not isinstance(motifs, (list, tuple)):
        motifs = [motifs]
    rows = []
    for m in motifs:
        df = sites_frame(m)
        if passing_only:
            df = df[df["passing"]]
        else:
            df = df[np.isfinite(df["score"])]
        for r in df.itertuples(index=False):
            rows.append((r.chrom, int(r.site_start), int(r.site_end), f"{m.name}|{r.name}", float(r.score), r.strand))
    if sort:
        rows.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    with open(path, "w") as fh:
        for chrom, start, end, name, score, strand in rows:
            fh.write(f"{chrom}\t{start}\t{end}\t{name}\t{score:.3f}\t{strand}\n")
    return len(rows)


# --- annotation against a motif database ---------------------------------


def _column_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlation of two (4, k) matrices -> (k,)."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (a * b).sum(axis=0) / denom, 0.0)
    return r


def compare_pwms(query: np.ndarray, target: np.ndarray, min_overlap: int = 5) -> tuple[float, int, str, int]:
    """Best alignment of `query` to `target` (both (4, w) probability
    matrices) in Tomtom's Pearson flavour: the score of an alignment is
    the *sum* of per-column Pearson correlations over the overlapping
    columns (so longer, consistently similar overlaps score higher, and a
    perfect match scores its width), maximized over all offsets with at
    least `min_overlap` overlapping columns and over both orientations
    of the query. Returns `(score, offset, orientation, overlap)` where
    `offset` is the target column the query's first column aligns to
    (negative: the query starts before the target) and orientation is
    "+" or "-" (query reverse-complemented).
    """
    query = np.asarray(query, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    wq, wt = query.shape[1], target.shape[1]
    min_overlap = max(1, min(min_overlap, wq, wt))
    best = (-np.inf, 0, "+", 0)
    for orientation, q in (("+", query), ("-", query[::-1, ::-1])):
        for offset in range(-(wq - min_overlap), wt - min_overlap + 1):
            q_start, t_start = max(0, -offset), max(0, offset)
            overlap = min(wq - q_start, wt - t_start)
            if overlap < min_overlap:
                continue
            score = float(_column_pearson(q[:, q_start : q_start + overlap], target[:, t_start : t_start + overlap]).sum())
            if score > best[0]:
                best = (score, offset, orientation, overlap)
    return best


def annotate(motifs, database, top: int = 3, min_overlap: int = 5):
    """Rank every motif in a MEME-format `database` (a path, or an
    already-parsed `{name: MemeMotif}` dict) against each discovered
    motif by `compare_pwms`, and return the `top` hits per motif as a
    DataFrame (query, rank, target, score, mean per-column correlation,
    offset, orientation, overlap).

    This is an in-process *ranking* in Tomtom's Pearson metric, not
    Tomtom: no p-value/E-value (which needs Tomtom's null distribution
    over the whole database). Use it to name what `fit` found -- e.g.
    against HOCOMOCO v12 core (`H12CORE_meme_format.meme`) -- and keep
    `scripts/validate_streme.py` (real Tomtom) for validation claims.
    """
    pd = _pandas()
    if not isinstance(motifs, (list, tuple)):
        motifs = [motifs]
    db = database if isinstance(database, dict) else read_meme(database)
    rows = []
    for m in motifs:
        query_name = getattr(m, "name", None) or getattr(m, "consensus", "query")
        pwm = m.pwm if not isinstance(m, MemeMotif) else m.pwm
        hits = []
        for target_name, target in db.items():
            score, offset, orientation, overlap = compare_pwms(pwm, target.pwm, min_overlap=min_overlap)
            hits.append((score, target_name, offset, orientation, overlap))
        hits.sort(key=lambda h: -h[0])
        for rank, (score, target_name, offset, orientation, overlap) in enumerate(hits[:top], start=1):
            rows.append(
                {
                    "query": query_name,
                    "rank": rank,
                    "target": target_name,
                    "score": score,
                    "mean_r": score / overlap if overlap else np.nan,
                    "offset": offset,
                    "orientation": orientation,
                    "overlap": overlap,
                }
            )
    return pd.DataFrame(rows)
