"""Plots (Phase 5): smoke tests that the figures draw and carry the right
labels; visual judgement is the notebook's job."""

from __future__ import annotations

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
pytest.importorskip("logomaker")

from pystreme import plot  # noqa: E402
from pystreme.discovery import MotifDiscovery  # noqa: E402

from test_discovery import _FIT_KW, _two_motif_store  # noqa: E402


@pytest.fixture(scope="module")
def motifs():
    disc = MotifDiscovery(_two_motif_store(n=150), holdout_frac=0.2, seed=0)
    return disc.fit(n_motifs=2, **_FIT_KW)


def test_logo_draws_information_content(motifs):
    ax = plot.logo(motifs[0])
    assert ax.get_title() == motifs[0].name
    assert ax.get_ylim() == (0, 2)
    assert len(ax.get_xticks()) == motifs[0].width
    # a bare array works too
    ax2 = plot.logo(np.full((4, 6), 0.25), title="flat")
    assert ax2.get_title() == "flat"


def test_positions_marks_the_central_window(motifs):
    m = motifs[0]
    ax = plot.positions(m)
    assert "central log p" in ax.get_title()
    spans = [p for p in ax.patches if p.get_alpha() == 0.15]
    assert len(spans) == 1
    x0, x1 = spans[0].get_x(), spans[0].get_x() + spans[0].get_width()
    assert (x0, x1) == pytest.approx((-m.central.half_width, m.central.half_width))


def test_positions_does_not_shade_without_central_signal(motifs):
    from dataclasses import replace
    from pystreme.statistics import CentralEnrichment

    m = motifs[0]
    flat = replace(m, central=CentralEnrichment(log_pvalue=-0.5, half_width=90.0, n_central=10, n_sites=10, expected=9.0),
                   central_logp=-0.5)
    ax = plot.positions(flat)
    assert not [p for p in ax.patches if p.get_alpha() == 0.15]


def test_report_one_row_per_motif(motifs):
    fig = plot.report(motifs)
    assert len(fig.axes) == 2 * len(motifs)
    mpl.pyplot.close(fig)


def test_report_orders_by_pvalue_and_labels_matches(motifs):
    from pystreme.results import sort_by_pvalue

    reversed_in = list(reversed(sort_by_pvalue(motifs)))  # least significant first
    labels = {m.name: f"DB{i}" for i, m in enumerate(motifs)}
    fig = plot.report(reversed_in, annotation=labels)
    titles = [ax.get_title() for ax in fig.axes[::2]]
    expected = sort_by_pvalue(motifs)
    assert [t.split()[0] for t in titles] == [m.name for m in expected]
    assert all(labels[m.name] in t for m, t in zip(expected, titles))
    mpl.pyplot.close(fig)

    fig = plot.report(reversed_in, order="round")
    assert [ax.get_title().split()[0] for ax in fig.axes[::2]] == [m.name for m in reversed_in]
    mpl.pyplot.close(fig)


def test_report_accepts_annotate_frame(motifs):
    from pystreme.results import annotate

    db = {"HIT.H12CORE.0.P.A": _fake_db_motif(motifs[0])}
    ann = annotate(motifs, db, top=1)
    fig = plot.report(motifs, annotation=ann)
    assert any("HIT (r=" in ax.get_title() for ax in fig.axes[::2])
    mpl.pyplot.close(fig)


def _fake_db_motif(m):
    from pystreme.meme_io import MemeMotif

    return MemeMotif(name="HIT", pwm=m.pwm.copy())
