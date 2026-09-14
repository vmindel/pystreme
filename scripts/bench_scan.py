#!/usr/bin/env python
"""Scan-kernel wall-clock benchmark (DESIGNDOC.md §3.1: "two scan kernels,
benchmarked not assumed"; L8's scanner half).

Times `scanner.scan` for both kernels on a synthetic 7232 x 200 bp store
(the size of the real SP1 peak set) with 40 PWMs of widths 6-15, on CPU
and, when visible, CUDA. Not a pytest test -- run it on a GPU node:

    python scripts/bench_scan.py

Record results in BENCHMARKS.md. The full-pipeline number (fit vs STREME)
is scripts/validate_streme.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pystreme.scanner import scan
from pystreme.sequence_store import SequenceStore


def _random_store(n: int, length: int, device: str, seed: int = 0) -> SequenceStore:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=(n, length), dtype=np.uint8)
    codes_t = torch.from_numpy(codes).to(device)
    store = SequenceStore.from_sequences(["A" * length], device=device)  # shape template
    store = SequenceStore(codes_t, torch.ones_like(codes_t, dtype=torch.bool), [store.intervals[0]] * n, [], device=device)
    store.fit_background(order=2)
    return store


def _random_pwms(n: int, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = 6 + i % 10
        raw = rng.random((4, w)) + 0.05
        out.append(raw / raw.sum(axis=0, keepdims=True))
    return out


def _time(fn, device: str, repeats: int) -> float:
    fn()  # warm-up (cuDNN autotune, allocator)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=7232)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n-pwms", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    pwms = _random_pwms(args.n_pwms)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"| device | kernel | {args.n_pwms} PWMs x {args.n_seqs} x {args.length} bp, both strands | per PWM |")
    print("|---|---|---|---|")
    for device in devices:
        store = _random_store(args.n_seqs, args.length, device)
        for kernel in ("gather", "conv"):
            t = _time(lambda: scan(pwms, store, revcomp=True, kernel=kernel), device, args.repeats)
            name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
            print(f"| {name} | {kernel} | {t * 1000:.0f} ms | {t / args.n_pwms * 1000:.2f} ms |")


if __name__ == "__main__":
    main()
