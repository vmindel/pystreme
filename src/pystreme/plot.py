"""Logos and positional histograms for discovered motifs. DESIGNDOC.md
Phase 5 ("logo plotting via logomaker; positional histogram plots").

matplotlib and logomaker are imported lazily: the core package never needs
them. Every function returns the matplotlib Axes it drew on so figures can
be composed further.
"""

from __future__ import annotations

import numpy as np

from .refine import column_information

_ALPHABET = "ACGT"


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("pystreme.plot needs matplotlib (`pip install matplotlib logomaker`)") from exc
    return plt


def logo(motif, ax=None, title: str | None = None):
    """Information-content sequence logo of `motif.pwm` (a `Motif`, a
    `meme_io.MemeMotif`, or a bare (4, w) array) via logomaker."""
    plt = _mpl()
    try:
        import logomaker
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("pystreme.plot.logo needs logomaker and pandas") from exc

    pwm = np.asarray(getattr(motif, "pwm", motif), dtype=np.float64)
    ic = column_information(pwm)  # bits per column
    heights = pd.DataFrame((pwm * ic[None, :]).T, columns=list(_ALPHABET))  # (w, 4): each base's share of the column's bits
    if ax is None:
        _, ax = plt.subplots(figsize=(max(2.0, 0.45 * pwm.shape[1]), 1.6))
    logomaker.Logo(heights, ax=ax, color_scheme="classic")
    ax.set_ylim(0, 2)
    ax.set_ylabel("bits")
    ax.set_xticks(range(pwm.shape[1]))
    ax.set_xticklabels(range(1, pwm.shape[1] + 1))
    ax.set_title(title if title is not None else getattr(motif, "name", ""))
    return ax


def positions(motif, ax=None, bins: int = 40, title: str | None = None, shade_below_logp: float = np.log(0.05)):
    """Histogram of the motif's passing-site centers relative to the
    sequence center (`Motif.positions`, §2.7), with the central window
    that won the CentriMo-style test shaded (only when that test is
    significant, `central_logp < shade_below_logp`) and its log p in the
    title. Sites at every offset with equal frequency means no positional
    signal; a spike at 0 is what a directly bound, well-summited factor
    looks like."""
    plt = _mpl()
    if motif.positions is None or len(motif.positions) == 0:
        raise ValueError("plot.positions: this Motif has no passing sites")
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2.2))
    ax.hist(motif.positions, bins=bins, color="0.35")
    ax.axvline(0, color="tab:red", lw=0.8)
    # shade the winning window only when the test says there is one: with
    # no central signal the "best" window is arbitrary (often nearly the
    # whole axis) and shading it would suggest a result that isn't there
    if motif.central is not None and motif.central.log_pvalue < shade_below_logp:
        h = motif.central.half_width
        ax.axvspan(-h, h, color="tab:red", alpha=0.15, lw=0)
    if title is None:
        title = motif.name
        if motif.central_logp is not None:
            title += f"   central log p = {motif.central_logp:.1f}"
    ax.set_title(title)
    ax.set_xlabel("site center relative to peak center (bp)")
    ax.set_ylabel("sites")
    return ax


def _labels(motifs, annotation) -> dict[str, str]:
    """`annotation` -> {motif name: label}. Accepts the DataFrame from
    `results.annotate` (rank-1 target per query, HOCOMOCO ids shortened to
    the factor name, with the mean per-column r) or a plain dict."""
    if annotation is None:
        return {}
    if isinstance(annotation, dict):
        return dict(annotation)
    best = annotation[annotation["rank"] == 1]
    out = {}
    for r in best.itertuples(index=False):
        target = str(r.target)
        if ".H12CORE" in target:
            target = target.split(".H12CORE")[0]
        out[str(r.query)] = f"{target} (r={r.mean_r:.2f})"
    return out


def report(motifs, figsize_per_row: tuple[float, float] = (9.0, 2.2), annotation=None, order: str = "pvalue"):
    """One row per motif: logo on the left, positional histogram on the
    right. Returns the matplotlib Figure.

    `annotation` puts a database match in each logo's title: the DataFrame
    from `results.annotate` (its rank-1 hit per motif) or a `{motif name:
    label}` dict. `order` is `"pvalue"` (default: most significant first,
    by `holdout_logp`, or `log_pvalue` where there is no hold-out) or
    `"round"` (discovery order, as `fit` returned them -- only roughly
    monotone in p, since each round is the best of what is *left* after
    erasing and the hold-out re-test is noisy). Names keep their round
    prefix either way.
    """
    from .results import sort_by_pvalue

    plt = _mpl()
    if not isinstance(motifs, (list, tuple)):
        motifs = [motifs]
    if order == "pvalue":
        motifs = sort_by_pvalue(motifs)
    elif order != "round":
        raise ValueError(f"report: order must be 'pvalue' or 'round', got {order!r}")
    labels = _labels(motifs, annotation)
    n = len(motifs)
    fig, axes = plt.subplots(n, 2, figsize=(figsize_per_row[0], figsize_per_row[1] * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for row, m in zip(axes, motifs):
        # two lines: identity (name, database match) over the numbers, so a
        # long name plus a label never runs past the logo axis
        head = m.name + (f"  ~ {labels[m.name]}" if m.name in labels else "")
        stats = f"n_sites={m.n_sites}" + (
            f"   hold-out log p={m.holdout_logp:.1f}" if m.holdout_logp is not None else "")
        logo(m, ax=row[0], title=f"{head}\n{stats}")
        row[0].title.set_fontsize("medium")
        if m.positions is not None and len(m.positions):
            positions(m, ax=row[1], title="")
            row[1].set_title(f"central log p = {m.central_logp:.1f}" if m.central_logp is not None else "")
        else:
            row[1].set_axis_off()
    fig.tight_layout()
    return fig
