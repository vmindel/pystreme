"""Golden outputs: `fit` on fixed synthetic datasets, frozen to JSON.

These pin end-to-end behaviour so that a change meant to be a pure
optimisation (scanner, seed counting, threshold search) can be checked
for having changed nothing. Each scenario fixes everything that feeds the
result: the data generator (defined here, not shared with other tests, so
editing another test cannot move these), the control (`DinucShuffle` with
an explicit seed and `engine="python"` -- the numba engine draws a
different shuffle from the same seed), the kernel, and every `fit`
argument.

Integers (widths, site counts, positions, strands, hold-out tables) and
strings must match exactly; floats to `RTOL`/`ATOL`, which absorbs
last-bit differences between BLAS/torch builds but not a real change.
A change that is *meant* to move results -- or one that only reorders
exact ties -- needs the fixture regenerated, and the diff reviewed:

    PYSTREME_UPDATE_GOLDEN=1 pytest tests/test_golden.py

CPU only: the CUDA kernels sum in a different order.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pystreme.control import DinucShuffle
from pystreme.discovery import MotifDiscovery
from pystreme.sequence_store import SequenceStore

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_fit.json"
UPDATE = os.environ.get("PYSTREME_UPDATE_GOLDEN") == "1"
RTOL = 1e-5
ATOL = 1e-5


def _plant(bases, rng, motif, pos):
    if rng.random() < 0.5:  # either strand
        motif = motif.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    bases[pos : pos + len(motif)] = list(motif)


def _two_motifs():
    """A central GC-box in 60% of sequences, an AP-1-like site anywhere in 40%."""
    rng = np.random.default_rng(101)
    a, b, length = "GGGGCGGGGC", "TGACTCATTT", 80
    seqs = []
    for _ in range(300):
        bases = list(rng.choice(list("ACGT"), length))
        if rng.random() < 0.6:
            _plant(bases, rng, a, 35 + int(rng.integers(-3, 4)))
        if rng.random() < 0.4:
            pos = int(rng.choice(np.r_[0:22, 48:70]))  # clear of the central site
            _plant(bases, rng, b, pos)
        seqs.append("".join(bases))
    return seqs


def _n_runs_odd_length():
    """Odd length, GC-skewed background, N runs and a degenerate motif."""
    rng = np.random.default_rng(202)
    length = 101
    seqs = []
    for _ in range(250):
        bases = list(rng.choice(list("ACGT"), length, p=[0.2, 0.3, 0.3, 0.2]))
        if rng.random() < 0.5:
            site = ["CACGTG"[i] if rng.random() < 0.9 else rng.choice(list("ACGT")) for i in range(6)]
            pos = int(rng.integers(0, length - 8))
            bases[pos : pos + 8] = ["T"] + site + ["A"]
        if rng.random() < 0.3:
            start, run = int(rng.integers(0, length - 12)), int(rng.integers(1, 12))
            bases[start : start + run] = ["N"] * run
        seqs.append("".join(bases))
    return seqs


def _forward_only():
    """A strand-specific site, only ever on the forward strand."""
    rng = np.random.default_rng(303)
    motif, length = "TTGCGCAAAT", 60
    seqs = []
    for _ in range(200):
        bases = list(rng.choice(list("ACGT"), length))
        if rng.random() < 0.5:
            pos = int(rng.integers(0, length - len(motif)))
            bases[pos : pos + len(motif)] = list(motif)
        seqs.append("".join(bases))
    return seqs


SCENARIOS = {
    "two_motifs_conv": dict(
        data=_two_motifs,
        disc=dict(widths=[8, 10], holdout_frac=0.2, seed=0),
        control_seed=11,
        fit=dict(n_motifs=2, bg_order=1, n_per_width=3, n_iter=10, kernel="conv"),
    ),
    "n_runs_order2_gather": dict(
        data=_n_runs_odd_length,
        disc=dict(widths=[6, 7, 8], holdout_frac=0.15, seed=5),
        control_seed=22,
        fit=dict(n_motifs=2, bg_order=2, n_per_width=4, n_iter=10, kernel="gather", keep_insignificant=True, patience=2),
    ),
    "forward_only_conv": dict(
        data=_forward_only,
        disc=dict(widths=[8, 10], holdout_frac=0.2, seed=3, revcomp=False),
        control_seed=33,
        fit=dict(n_motifs=1, bg_order=1, n_per_width=3, n_iter=10, kernel="conv", keep_insignificant=True),
    ),
}


def _float(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else ("-inf" if x < 0 else "inf")


def _snapshot(name: str) -> list[dict]:
    spec = SCENARIOS[name]
    store = SequenceStore.from_sequences(spec["data"]())
    control = DinucShuffle(seed=spec["control_seed"], engine="python")
    disc = MotifDiscovery(store, control=control, **spec["disc"])
    out = []
    for m in disc.fit(**spec["fit"]):
        h = m.holdout
        out.append(
            {
                "name": m.name,
                "consensus": m.consensus,
                "seed": m.seed,
                "width": m.width,
                "n_sites": m.n_sites,
                "threshold": _float(m.threshold),
                "log_pvalue": _float(m.log_pvalue),
                "holdout_logp": _float(m.holdout_logp),
                "holdout_table": None if h is None else list(h.table),
                "central_logp": _float(m.central_logp),
                "central_half_width": None if m.central is None else _float(m.central.half_width),
                "pwm": np.round(m.pwm, 8).tolist(),
                "positions": m.sites.positions.cpu().tolist(),
                "strands": m.sites.strands.cpu().tolist(),
                "scores": [_float(s) for s in m.sites.scores.cpu().tolist()],
            }
        )
    return out


def _load() -> dict:
    return json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}


def _close(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _compare(got: list[dict], want: list[dict], scenario: str):
    hint = f"[{scenario}] if this change is meant to move results, regenerate: PYSTREME_UPDATE_GOLDEN=1 pytest tests/test_golden.py"
    assert [m["name"] for m in got] == [m["name"] for m in want], hint
    for g, w in zip(got, want):
        where = f"{hint}\n  motif {w['name']}"
        for key in ("consensus", "seed", "width", "n_sites", "holdout_table", "positions", "strands"):
            assert g[key] == w[key], f"{where}: {key} differs"
        for key in ("threshold", "log_pvalue", "holdout_logp", "central_logp", "central_half_width"):
            assert _close(g[key], w[key]), f"{where}: {key} {g[key]} != {w[key]}"
        np.testing.assert_allclose(np.array(g["pwm"]), np.array(w["pwm"]), rtol=RTOL, atol=ATOL, err_msg=where)
        bad = [i for i, (a, b) in enumerate(zip(g["scores"], w["scores"])) if not _close(a, b)]
        assert len(g["scores"]) == len(w["scores"]) and not bad, f"{where}: scores differ at rows {bad[:10]}"


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_fit_matches_golden_output(scenario):
    got = _snapshot(scenario)
    if UPDATE:
        golden = _load()
        golden[scenario] = got
        GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"golden output for {scenario} regenerated")
    golden = _load()
    if scenario not in golden:
        pytest.fail(f"no golden output for {scenario}; generate it with PYSTREME_UPDATE_GOLDEN=1 pytest tests/test_golden.py")
    _compare(got, golden[scenario], scenario)


def test_golden_scenarios_still_find_what_was_planted():
    """The frozen numbers are only worth freezing if they are right: the
    planted motifs are recovered, in the scenarios built for it."""
    golden = _load()
    if not golden:
        pytest.skip("golden fixture not generated yet")
    rc = str.maketrans("ACGT", "TGCA")

    def matches(consensus, word):
        return word in consensus or word.translate(rc)[::-1] in consensus

    two = golden["two_motifs_conv"]
    assert len(two) == 2
    assert matches(two[0]["consensus"], "GGGCGGG") and two[0]["central_logp"] < math.log(1e-6)
    assert matches(two[1]["consensus"], "TGACTCA")
    assert all(m["holdout_logp"] < math.log(0.05) for m in two)
    assert matches(golden["n_runs_order2_gather"][0]["consensus"], "CACGTG")
    # forward orientation, not its RC, and aligned on the site: the core
    # alone used to pass with a seed matrix shifted one base right
    fwd = golden["forward_only_conv"][0]
    assert fwd["consensus"] == "TTGCGCAAAT"
    pwm = np.array(fwd["pwm"])
    assert not np.allclose(pwm.max(axis=0), 0.8), "an unrefined seed_to_pwm matrix, not one estimated from sites"


def test_golden_scenarios_are_deterministic():
    """Two runs in one process agree exactly -- otherwise a golden mismatch
    could be noise rather than a change."""
    assert _snapshot("forward_only_conv") == _snapshot("forward_only_conv")
