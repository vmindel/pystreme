"""Tables and annotation (DESIGNDOC.md Phase 5): summary / sites_frame /
annotate, and the convenience constructors."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pystreme.discovery import MotifDiscovery
from pystreme.meme_io import read_meme
from pystreme.results import annotate, compare_pwms, sites_frame, summary

from test_discovery import _FIT_KW, MOTIF_A, _matches, _two_motif_store, _argmax_word

pd = pytest.importorskip("pandas")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HOCOMOCO = os.environ.get("PYSTREME_TEST_HOCOMOCO", "")  # HOCOMOCO v12 core MEME file; tests using it skip when unset


@pytest.fixture(scope="module")
def fitted():
    disc = MotifDiscovery(_two_motif_store(n=200), holdout_frac=0.2, seed=0)
    return disc, disc.fit(n_motifs=2, **_FIT_KW)


def test_summary_one_row_per_motif(fitted):
    _, motifs = fitted
    df = summary(motifs)
    assert list(df["name"]) == [m.name for m in motifs]
    assert list(df["round"]) == [1, 2]
    assert (df["n_sites"] == [m.n_sites for m in motifs]).all()
    assert df["holdout_logp"].notna().all()
    assert set(df.columns) >= {"consensus", "width", "train_logp", "central_logp", "central_half_width", "threshold", "seed"}
    assert len(summary(motifs[0])) == 1  # a single Motif works too


def test_sites_frame_matches_motif_sites(fitted):
    disc, motifs = fitted
    m = motifs[0]
    df = m.sites_frame()
    assert len(df) == disc.store.n_seqs
    assert df["passing"].sum() == m.n_sites
    assert (df["site_end"] - df["site_start"])[df["passing"]].eq(m.width).all()
    # offsets of passing rows are exactly Motif.positions (same order)
    np.testing.assert_allclose(df.loc[df["passing"], "offset"].to_numpy(), m.positions)
    # site coordinates are interval.start + best position
    iv = m.intervals[0]
    assert df.loc[0, "site_start"] == iv.start + int(m.sites.positions[0])
    assert set(df["strand"]) <= {"+", "-"}


def test_sites_bed_writes_passing_sites_in_bed6(fitted, tmp_path):
    from pystreme.results import sites_bed

    disc, motifs = fitted
    path = tmp_path / "sites.bed"
    n = sites_bed(motifs, path)
    assert n == sum(m.n_sites for m in motifs)
    lines = [l.rstrip("\n").split("\t") for l in open(path)]
    assert len(lines) == n and all(len(l) == 6 for l in lines)
    chroms_starts = [(l[0], int(l[1])) for l in lines]
    assert chroms_starts == sorted(chroms_starts)
    names = {l[3].split("|")[0] for l in lines}
    assert names == {m.name for m in motifs}
    # each line's width is its motif's width and its strand is +/-
    widths = {m.name: m.width for m in motifs}
    assert all(int(l[2]) - int(l[1]) == widths[l[3].split("|")[0]] for l in lines)
    assert {l[5] for l in lines} <= {"+", "-"}
    # every peak's best window when passing_only=False; MotifDiscovery.to_bed delegates
    n_all = disc.to_bed(motifs, path, passing_only=False)
    assert n_all == len(motifs) * disc.store.n_seqs


def test_sites_frame_requires_intervals():
    from dataclasses import replace

    disc = MotifDiscovery(_two_motif_store(n=60), holdout_frac=0.0, seed=1)
    m = disc.discover(widths=[10], bg_order=1, n_per_width=2, n_iter=5, kernel="gather")
    assert len(sites_frame(m)) == 60
    with pytest.raises(ValueError, match="no intervals"):
        sites_frame(replace(m, intervals=None))


def test_compare_pwms_perfect_match_and_reverse_complement():
    sp1 = read_meme(os.path.join(FIXTURES, "MA0079.1_SP1.meme"))["MA0079.1"].pwm
    score, offset, orientation, overlap = compare_pwms(sp1, sp1)
    assert score == pytest.approx(sp1.shape[1])  # per-column r = 1, summed over every column
    assert (offset, orientation, overlap) == (0, "+", sp1.shape[1])

    rc = sp1[::-1, ::-1]
    score, offset, orientation, overlap = compare_pwms(rc, sp1)
    assert score == pytest.approx(sp1.shape[1])
    assert orientation == "-"

    # a shifted sub-motif aligns at the right offset
    sub = sp1[:, 2:8]
    score, offset, orientation, overlap = compare_pwms(sub, sp1)
    assert (offset, orientation, overlap) == (2, "+", 6)
    assert score == pytest.approx(6)


def test_annotate_names_the_implanted_motif(fitted):
    _, motifs = fitted
    db = {
        "SP1": read_meme(os.path.join(FIXTURES, "MA0079.1_SP1.meme"))["MA0079.1"],
        "KLF4": read_meme(os.path.join(FIXTURES, "MA0039.1_KLF4.meme"))["MA0039.1"],
    }
    df = annotate(motifs, db, top=2)
    assert list(df.columns) == ["query", "rank", "target", "score", "mean_r", "offset", "orientation", "overlap"]
    assert len(df) == 4
    first = df[(df["query"] == motifs[0].name) & (df["rank"] == 1)].iloc[0]
    assert _matches(_argmax_word(motifs[0].pwm), MOTIF_A)
    assert first["target"] == "SP1"  # GGGGCGGGGG is the SP1 GC-box
    assert first["mean_r"] > 0.8


@pytest.mark.skipif(not os.path.exists(HOCOMOCO), reason="HOCOMOCO v12 file not on this machine")
def test_hocomoco_file_parses_and_annotates():
    db = read_meme(HOCOMOCO)
    assert len(db) == 1443
    assert all(m.pwm.shape[0] == 4 and m.nsites for m in db.values())
    np.testing.assert_allclose(next(iter(db.values())).pwm.sum(axis=0), 1.0, atol=1e-6)
    sp1 = read_meme(os.path.join(FIXTURES, "MA0079.1_SP1.meme"))["MA0079.1"]
    hits = annotate(sp1, db, top=3)
    assert hits["target"].str.startswith("SP").any() or hits["target"].str.startswith("KLF").any(), hits


def test_convenience_constructors_fit_a_background():
    seqs = ["ACGT" * 20 for _ in range(5)]
    disc = MotifDiscovery.from_sequences(seqs, bg_order=1, widths=[6], seed=3)
    assert disc.store.bg_order == 1 and disc.widths == [6] and disc.seed == 3


def test_from_fasta_constructor(tmp_path):
    path = tmp_path / "seqs.fa"
    path.write_text("".join(f">s{i}\n{'ACGT' * 15}\n" for i in range(4)))
    disc = MotifDiscovery.from_fasta(path, bg_order=0)
    assert disc.store.n_seqs == 4 and disc.store.bg_order == 0
