"""Site calling with moodspy (`pystreme.sites`, `disc.scan`, `disc.to_bed`),
checked against a brute-force scan of the same matrix."""

from __future__ import annotations

import numpy as np
import pytest

from pystreme.discovery import MotifDiscovery
from pystreme.scanner import log_probabilities
from pystreme.sequence_store import SequenceStore
from pystreme.sites import find_sites, store_background, write_sites_bed

mp = pytest.importorskip("moodspy")
pytest.importorskip("pandas")

_COMP = str.maketrans("ACGT", "TGCA")


def _random_seqs(n, length, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]


def _random_pwm(width, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((4, width)) + 0.05
    return raw / raw.sum(axis=0, keepdims=True)


def _brute(pwm, seqs, bg, threshold):
    """Every (row, start, strand, score) with score >= threshold, in bits."""
    mat = log_probabilities(pwm) / np.log(2.0) - np.log2(bg)[:, None]
    w = mat.shape[1]
    idx = {b: i for i, b in enumerate("ACGT")}
    out = []
    for r, s in enumerate(seqs):
        for i in range(len(s) - w + 1):
            word = s[i : i + w]
            if "N" in word:
                continue
            fwd = sum(mat[idx[b], j] for j, b in enumerate(word))
            rc = sum(mat[idx[b], j] for j, b in enumerate(word.translate(_COMP)[::-1]))
            for strand, score in (("+", fwd), ("-", rc)):
                if score >= threshold:
                    out.append((r, i, strand, score))
    return out


@pytest.mark.parametrize("p_cutoff", [1.0, 0.05, 1e-3])
def test_sites_match_brute_force(p_cutoff):
    seqs = _random_seqs(30, 60, seed=0)
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=2)
    pwm = _random_pwm(7, seed=1)
    bg = store_background(store)
    mat = log_probabilities(pwm) / np.log(2.0) - np.log2(bg)[:, None]
    thr = mp.threshold_from_p(mat, p_cutoff, bg)

    sites = find_sites(pwm, store, p_cutoff=p_cutoff, overlaps=None)
    expected = _brute(pwm, seqs, bg, thr)
    assert len(sites) == len(expected)
    got = {(r, s, st): sc for r, s, st, sc in zip(sites["row"], sites["start"], sites["strand"], sites["score"])}
    for r, i, strand, score in expected:
        assert got[(r, i, strand)] == pytest.approx(score, abs=1e-9)


def test_default_background_is_the_store_marginals():
    store = SequenceStore.from_sequences(_random_seqs(20, 50, seed=2))
    # no fitted background: the store's own composition
    counts = np.bincount(store.codes.flatten().numpy(), minlength=4)[:4] + 1.0
    np.testing.assert_allclose(store_background(store), counts / counts.sum())
    store.fit_background(order=1)
    np.testing.assert_allclose(store_background(store), store.bg_marginals.exp().double().numpy())


def test_planted_sites_coordinates_strand_and_offset():
    word = "GGGGCGGGGC"
    seqs = _random_seqs(4, 101, seed=4)
    seqs[1] = seqs[1][:20] + word + seqs[1][30:]
    seqs[2] = seqs[2][:45] + word.translate(_COMP)[::-1] + seqs[2][55:]
    store = SequenceStore.from_sequences(seqs)
    pwm = np.full((4, len(word)), 0.01)
    for j, b in enumerate(word):
        pwm["ACGT".index(b), j] = 0.97
    sites = find_sites({"SP": pwm}, store, p_cutoff=1e-5, background=np.full(4, 0.25))
    assert list(sites["motif"].unique()) == ["SP"]
    one = sites[sites["row"] == 1].iloc[0]
    assert (one["start"], one["end"], one["strand"]) == (20, 30, "+")
    assert one["offset"] == pytest.approx(20 + 4.5 - 50)  # site centre minus window centre
    two = sites[sites["row"] == 2].iloc[0]
    assert (two["start"], two["strand"], two["offset"]) == (45, "-", -0.5)
    assert one["peak"] == "seq_1" and one["chrom"] == "control"


def test_n_and_erased_bases_are_never_inside_a_site():
    seqs = _random_seqs(10, 80, seed=5)
    seqs[0] = seqs[0][:30] + "NNN" + seqs[0][33:]
    store = SequenceStore.from_sequences(seqs)
    store.erase(np.array([1]), np.array([10]), 20)
    sites = find_sites(_random_pwm(6, seed=6), store, p_cutoff=1.0, overlaps=None)
    s0 = sites[sites["row"] == 0]
    assert not ((s0["start"] < 33) & (s0["end"] > 30)).any()
    s1 = sites[sites["row"] == 1]
    assert not ((s1["start"] < 30) & (s1["end"] > 10)).any()
    assert len(sites) == 2 * (75 * 10 - 8 - 25)  # 75 windows per row; the N run costs 8, the erased 20 bp cost 25


def test_idx_subset_scores_like_the_full_scan():
    seqs = _random_seqs(40, 60, seed=7)
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=2)
    pwm = _random_pwm(8, seed=8)
    full = find_sites(pwm, store, p_cutoff=0.01)
    idx = np.array([3, 17, 25])
    sub = find_sites(pwm, store, p_cutoff=0.01, idx=idx)
    assert set(sub["row"]) <= set(idx)
    want = full[full["row"].isin(idx)].reset_index(drop=True)
    np.testing.assert_array_equal(sub[["row", "start"]].to_numpy(), want[["row", "start"]].to_numpy())
    np.testing.assert_allclose(sub["score"], want["score"])


def test_disc_scan_and_to_bed(tmp_path):
    seqs = _random_seqs(25, 70, seed=9)
    disc = MotifDiscovery.from_sequences(seqs, bg_order=1)
    pwms = {"a": _random_pwm(6, seed=10), "b": _random_pwm(9, seed=11)}
    sites = disc.scan(pwms, p_cutoff=0.01)
    assert list(sites["motif"].cat.categories) == ["a", "b"]
    path = tmp_path / "sites.bed"
    n = disc.to_bed(pwms, path, p_cutoff=0.01)
    assert n == len(sites)
    lines = [l.rstrip("\n").split("\t") for l in open(path)]
    assert len(lines) == n and all(len(l) == 6 for l in lines)
    assert [(l[0], int(l[1])) for l in lines] == sorted((l[0], int(l[1])) for l in lines)
    assert {l[3].split("|")[0] for l in lines} == {"a", "b"}
    assert write_sites_bed(sites, tmp_path / "unsorted.bed", sort=False) == n


def _overlapping_pairs(df, opposite_only):
    out = []
    for _, g in df.groupby(["motif", "row"], observed=True):
        g = g.sort_values("start")
        s, e, st, sc = (g[c].to_numpy() for c in ("start", "end", "strand", "score"))
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if s[j] >= e[i]:
                    break
                if not opposite_only or st[i] != st[j]:
                    out.append((sc[i], sc[j]))
    return out


@pytest.mark.parametrize("overlaps", ["any", "opposite"])
def test_overlaps_are_collapsed_to_the_best_hit(overlaps):
    # an exact palindrome (matches one site on both strands) and a tandem
    # GC-box (copies 5 bp apart overlap on the same strand)
    seqs = _random_seqs(40, 80, seed=12)
    seqs = [s[:10] + "GTCACGTGAC" + s[20:40] + "GGGGCGGGGCGGGGCGGGGC" + s[60:] for s in seqs]
    store = SequenceStore.from_sequences(seqs)

    def sharp(word):
        m = np.full((4, len(word)), 0.03)
        for j, b in enumerate(word):
            m["ACGT".index(b), j] = 0.91
        return m

    pwm = {"ebox": sharp("GTCACGTGAC"), "gc": sharp("GGGGCGGGGC")}
    raw = find_sites(pwm, store, p_cutoff=1e-3, overlaps=None)
    kept = find_sites(pwm, store, p_cutoff=1e-3, overlaps=overlaps)
    assert len(kept) < len(raw)
    assert _overlapping_pairs(raw, opposite_only=(overlaps == "opposite"))
    assert not _overlapping_pairs(kept, opposite_only=(overlaps == "opposite"))
    # every peak with a hit keeps its best one
    best_raw = raw.groupby(["motif", "row"], observed=True)["score"].max()
    best_kept = kept.groupby(["motif", "row"], observed=True)["score"].max()
    pd = pytest.importorskip("pandas")
    pd.testing.assert_series_equal(best_raw, best_kept)
    # nothing is dropped unless it overlaps a kept hit that scores at least as well
    k = set(zip(kept["row"], kept["start"], kept["strand"]))
    for r in raw.itertuples(index=False):
        if (r.row, r.start, r.strand) in k:
            continue
        same = kept[(kept["motif"] == r.motif) & (kept["row"] == r.row) & (kept["start"] < r.end) & (kept["end"] > r.start)]
        if overlaps == "opposite":
            same = same[same["strand"] != r.strand]
        assert (same["score"] >= r.score - 1e-12).any()
    with pytest.raises(ValueError, match="overlaps"):
        find_sites(pwm, store, overlaps="both")
