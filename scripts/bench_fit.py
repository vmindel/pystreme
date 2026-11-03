#!/usr/bin/env python
"""End-to-end `fit` benchmark with a per-component time breakdown.

A fixed synthetic peak set (two planted motifs, `--n` x `--length` bp),
the default width range 6-15, and cProfile cumulative times for the parts
that dominate: seed counting, scanning, the threshold search, shuffling.
Use it before and after a change meant to speed things up, on the same
machine, and check `tests/test_golden.py` for the change not moving any
result:

    python scripts/bench_fit.py                       # CPU, conv kernel
    python scripts/bench_fit.py --device cuda --kernel gather
    python scripts/bench_fit.py --json before.json    # keep the numbers

Times on a shared node vary by 10-20% between runs; `--repeat` takes the
fastest.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
import warnings

import numpy as np
import torch

from pystreme import MotifDiscovery
from pystreme.control import DinucShuffle
from pystreme.sequence_store import SequenceStore

_COMPONENTS = {
    "top_seeds": "seeds.py:.*top_seeds",
    "scan": "scanner.py:.*scan$",
    "optimal_threshold": "statistics.py:.*optimal_threshold",
    "refine": "refine.py:.*refine$",
    "dinuc_shuffle": "control.py:.*generate",
}


def _peaks(n: int, length: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    a, b = "GGGGCGGGGC", "TGACTCATTT"
    centre = (length - len(a)) // 2
    seqs = []
    for _ in range(n):
        bases = list(rng.choice(list("ACGT"), length))
        if rng.random() < 0.6:
            pos = centre + int(rng.integers(-5, 6))
            bases[pos : pos + len(a)] = list(a)
        if rng.random() < 0.4:
            pos = int(rng.integers(0, length - len(b)))
            if pos + len(b) <= centre - 5 or pos >= centre + 5 + len(a):
                bases[pos : pos + len(b)] = list(b)
        seqs.append("".join(bases))
    return seqs


def _component_times(profile: cProfile.Profile) -> dict[str, float]:
    import re

    stats = pstats.Stats(profile).stats
    out = {}
    for name, pattern in _COMPONENTS.items():
        rx = re.compile(pattern)
        # cumulative time of the outermost matching frames only
        out[name] = sum(
            ct for (fn, line, func), (cc, nc, tt, ct, callers) in stats.items()
            if rx.search(f"{fn.split('/')[-1]}:{line}({func})".replace("(", ":").rstrip(")"))
            and not any(rx.search(f"{c[0].split('/')[-1]}:{c[1]}:{c[2]}") for c in callers)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-motifs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kernel", default="auto")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads; 0 leaves the default")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    warnings.simplefilter("ignore")

    store = SequenceStore.from_sequences(_peaks(args.n, args.length), device=args.device)
    store.fit_background(order=2)
    best = None
    for _ in range(args.repeat):
        disc = MotifDiscovery(store, control=DinucShuffle(seed=1, engine="python"), holdout_frac=0.1, seed=0)
        profile = cProfile.Profile()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        profile.enable()
        motifs = disc.fit(n_motifs=args.n_motifs, kernel=args.kernel)
        profile.disable()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        if best is None or wall < best["wall_s"]:
            best = {
                "wall_s": wall,
                "components_s": _component_times(profile),
                "motifs": [m.name for m in motifs],
            }
    best.update(
        n=args.n, length=args.length, device=args.device, kernel=args.kernel, torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
    print(f"fit: {best['wall_s']:.2f} s  ({args.n} x {args.length} bp, {args.device}, kernel={args.kernel})")
    for name, t in sorted(best["components_s"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18} {t:7.2f} s  {100 * t / best['wall_s']:5.1f}%")
    print("  motifs:", ", ".join(best["motifs"]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(best, fh, indent=1)


if __name__ == "__main__":
    main()
