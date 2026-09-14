#!/usr/bin/env python
"""L1: the scanner against FIMO on real peaks (DESIGNDOC.md §4).

Same sequences, same PWMs, same order-0 background for both tools; compare
each sequence's best site (over positions and both strands): Pearson r on
the best scores (FIMO reports bits with two-decimal rounding, pystreme
nats -- r is scale-free) and the fraction of sequences with the identical
argmax position and strand. Requires the MEME suite (`fimo` on PATH):

    python scripts/validate_fimo.py --peaks peaks.bed --genome genome.fa \\
        --n-peaks 2000 --out validation/l1

Only order 0 is comparable: FIMO converts the PWM to log-odds with order-0
frequencies whatever the background file's order. The order-2 cumsum
term is covered by the naive-reference tests in tests/test_sequence_store.py.

PWMs are smoothed here (`--pseudo`, uniform, then renormalized) and FIMO
is run with `--motif-pseudo 0` so neither tool applies its own different
zero-handling to JASPAR's exact-zero cells.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from pystreme.meme_io import MemeMotif, read_meme, write_meme
from pystreme.scanner import scan
from pystreme.sequence_store import SequenceStore
from validate_streme import FIXTURES, _subset_bed


def _fimo_scores(fimo_tsv: Path, motif_name: str):
    """All (sequence, start, strand) -> score for one motif, plus the best per sequence."""
    scores: dict[tuple[str, int, str], float] = {}
    best: dict[str, tuple[float, int, str]] = {}
    with open(fimo_tsv) as fh:
        header = None
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            row = dict(zip(header, fields))
            if row["motif_id"] != motif_name:
                continue
            score = float(row["score"])
            seq, start, strand = row["sequence_name"], int(row["start"]) - 1, row["strand"]
            scores[(seq, start, strand)] = score
            cur = best.get(seq)
            # tie-break like the scanner: earliest position, + before -
            if cur is None or score > cur[0] or (score == cur[0] and (start, strand) < (cur[1], cur[2])):
                best[seq] = (score, start, strand)
    return scores, best


def _symmetrize_background(store: SequenceStore) -> np.ndarray:
    """MEME tools average each base's frequency with its complement's when
    scanning both strands; do the same to pystreme's order-0 table so the
    two scanners see one background. Returns the ACGT frequencies."""
    freq = store.bg_table.exp()
    sym = (freq + freq.flip(0)) / 2  # A<->T, C<->G
    store.bg_table = sym.log()
    codes = store.codes.long().clamp(max=3)
    bg_ll = torch.where(store.mask, store.bg_table[codes], torch.zeros(codes.shape, dtype=sym.dtype))
    zeros = torch.zeros(store.n_seqs, 1, dtype=bg_ll.dtype)
    store.bg_cumsum = torch.cat([zeros, torch.cumsum(bg_ll, dim=1)], dim=1)
    return sym.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True, help="summit-centred peak BED")
    ap.add_argument("--genome", required=True, help="genome FASTA, indexed (.fai alongside)")
    ap.add_argument("--n-peaks", type=int, default=2000)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--pseudo", type=float, default=0.01)
    ap.add_argument("--out", default="validation/l1")
    args = ap.parse_args()
    if shutil.which("fimo") is None:
        raise SystemExit("fimo not on PATH -- `module load MEME/5.5.0-GCC-10.3.0` first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bed = out / "peaks.bed"
    _subset_bed(args.peaks, bed, args.n_peaks)
    store = SequenceStore.from_bed(bed, args.genome, size=args.size)
    store.fit_background(order=0)
    fasta = out / "primary.fa"
    store.to_fasta(fasta)
    names = [f"{iv.chrom}:{iv.start}-{iv.end}" for iv in store.intervals]
    assert len(set(names)) == len(names), "duplicate peak windows; FIMO needs unique names"

    bg = _symmetrize_background(store)
    bgfile = out / "bg0.txt"
    bgfile.write_text("# order 0\n" + "".join(f"{b} {p:.6f}\n" for b, p in zip("ACGT", bg)))

    motifs = {}
    for fx in sorted(Path(FIXTURES).glob("*.meme")):
        for name, m in read_meme(fx).items():
            pwm = m.pwm + args.pseudo
            motifs[name] = pwm / pwm.sum(axis=0, keepdims=True)
    meme = out / "motifs.meme"
    write_meme([MemeMotif(name=n, pwm=p) for n, p in motifs.items()], meme, background=bg)

    fimo_tsv = out / "fimo.tsv"
    t0 = time.perf_counter()
    with open(fimo_tsv, "w") as fh:
        subprocess.run(
            ["fimo", "--text", "--thresh", "1.0", "--motif-pseudo", "0", "--bgfile", str(bgfile), str(meme), str(fasta)],
            stdout=fh, stderr=open(out / "fimo.log", "w"), check=True,
        )
    t_fimo = time.perf_counter() - t0

    lines = [f"# L1: pystreme scanner vs FIMO on {store.n_seqs} peaks x {store.length} bp, order-0 background", "",
             f"- FIMO (`--text --thresh 1.0`, both strands, all positions): {t_fimo:.1f}s", "",
             "| motif | kernel | n | Pearson r (best scores) | same position | same position+strand | same or tied (<= 0.01 bit) | pystreme scan |",
             "|---|---|---|---|---|---|---|---|"]
    ok = True
    for name, pwm in motifs.items():
        fimo_all, fimo_best = _fimo_scores(fimo_tsv, name)
        for kernel in ("gather", "conv"):
            t0 = time.perf_counter()
            res = scan([pwm], store, revcomp=True, kernel=kernel)
            t_scan = time.perf_counter() - t0
            ours = res.scores[0].numpy(), res.positions[0].numpy(), res.strands[0].numpy()
            f_scores, o_scores, same_pos_l, same_both_l, tied_l = [], [], [], [], []
            for i, nm in enumerate(names):
                if nm not in fimo_best or not np.isfinite(ours[0][i]):
                    continue
                s, p, st = fimo_best[nm]
                o_p, o_st = int(ours[1][i]), "+" if ours[2][i] > 0 else "-"
                f_scores.append(s); o_scores.append(ours[0][i])
                same_pos_l.append(p == o_p)
                same_both_l.append(p == o_p and st == o_st)
                # a disagreement counts as a tie when FIMO's own score at
                # pystreme's pick is within its 0.01-bit output resolution of its best
                tied_l.append(p == o_p or s - fimo_all.get((nm, o_p, o_st), -np.inf) <= 0.01 + 1e-9)
            f_scores, o_scores = np.array(f_scores), np.array(o_scores)
            r = float(np.corrcoef(f_scores, o_scores)[0, 1])
            same_pos, same_both, tied = (float(np.mean(x)) for x in (same_pos_l, same_both_l, tied_l))
            ok &= r > 0.999 and tied >= 0.99
            lines.append(f"| {name} | {kernel} | {len(f_scores)} | {r:.5f} | {same_pos:.3%} | {same_both:.3%} | {tied:.3%} | {t_scan:.2f}s |")
    lines += ["", f"**L1 {'PASS' if ok else 'FAIL'}** (criteria: r > 0.999, identical argmax -- or tied within FIMO's score resolution -- on >= 99%)"]
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
