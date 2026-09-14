#!/usr/bin/env python
"""L6/L8: head-to-head against the real STREME on a real peak set.

Not a pytest test -- this is the go/no-go for "reliable in the sense that
matters" (DESIGNDOC.md §4 L6) plus the wall-clock number that justifies the
project (L8). It needs the MEME suite (`streme` and `tomtom` on PATH) and a
real genome, and belongs on a GPU node for the timing to mean anything:

    python scripts/validate_streme.py --peaks peaks.bed --genome genome.fa \\
        --n-peaks 2000 --device cuda --out validation/run

What it does:
  1. extracts the peaks at --size (same SequenceStore both tools will see),
     writes them as FASTA so STREME gets *identical* input;
  2. runs `disc.fit(n_motifs=N)` (timed, in-process) and writes pystreme.meme;
  3. runs `streme --p primary.fa --dna --nmotifs N` (timed, including process
     startup) on the same FASTA, letting STREME build its own shuffled
     control, the same way pystreme's default does;
  4. runs Tomtom pystreme-vs-STREME (do the two tools find the same motifs?)
     and both-vs-JASPAR SP1/KLF4 (does either find the *right* motif?);
  5. prints a summary and writes it to <out>/summary.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pystreme.discovery import MotifDiscovery
from pystreme.meme_io import read_meme

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _subset_bed(src, dst, n):
    with open(src) as fh, open(dst, "w") as out:
        for i, line in enumerate(fh):
            if n is not None and i >= n:
                break
            out.write(line)


def _run(cmd, log):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - t0
    Path(log).write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return elapsed


def _tomtom(query, target, out_tsv):
    # -text: TSV on stdout, no HTML step (tomtom's tomtom_xml_to_html
    # helper exits abnormally on this cluster's module even when the
    # comparison itself succeeded, which made the -oc form look failed)
    out_tsv = Path(out_tsv)
    proc = subprocess.run(
        ["tomtom", "-no-ssc", "-text", "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10",
         str(query), str(target)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    out_tsv.write_text(proc.stdout)
    out_tsv.with_suffix(".log").write_text(proc.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        print(proc.stderr[-3000:])
        raise SystemExit(f"tomtom failed ({proc.returncode}) on {query} vs {target}")
    rows = []
    header = None
    for line in proc.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if header is None:
            header = fields
            continue
        rows.append(dict(zip(header, fields)))
    best = {}
    for r in rows:
        q = r["Query_ID"]
        if q not in best or float(r["E-value"]) < float(best[q]["E-value"]):
            best[q] = r
    return best


def _meme_summary(path):
    out = []
    for name, m in read_meme(path).items():
        ic = float((m.pwm * np.log2(np.clip(m.pwm, 1e-9, 1) / 0.25)).sum())
        out.append((name, m.width, m.nsites, m.evalue, ic))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True, help="summit-centred peak BED")
    ap.add_argument("--genome", required=True, help="genome FASTA, indexed (.fai alongside)")
    ap.add_argument("--n-peaks", type=int, default=None, help="use only the first N peaks (default: all)")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--n-motifs", type=int, default=3)
    ap.add_argument("--minw", type=int, default=6)
    ap.add_argument("--maxw", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kernel", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="validation/run")
    ap.add_argument("--skip-streme", action="store_true", help="only run pystreme + Tomtom vs JASPAR")
    ap.add_argument("--annotate", default=None, help="MEME-format motif database for pystreme.results.annotate (e.g. HOCOMOCO)")
    ap.add_argument("--report", action="store_true", help="also save plot.report(motifs) as <out>/report.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tool in ("streme", "tomtom"):
        if not args.skip_streme and shutil.which(tool) is None:
            raise SystemExit(f"{tool} not on PATH -- `module load MEME/5.5.0-GCC-10.3.0` first")
    have_tomtom = shutil.which("tomtom") is not None  # with --skip-streme, Tomtom vs JASPAR is optional

    bed = out / "peaks.bed"
    _subset_bed(args.peaks, bed, args.n_peaks)

    # 1. one extraction, shared by both tools
    t0 = time.perf_counter()
    disc = MotifDiscovery.from_bed(bed, args.genome, size=args.size, bg_order=2, device=args.device,
                                   widths=range(args.minw, args.maxw + 1), seed=args.seed)
    t_extract = time.perf_counter() - t0
    fasta = out / "primary.fa"
    disc.store.to_fasta(fasta)
    print(f"extracted {disc.store.n_seqs} x {disc.store.length} bp in {t_extract:.1f}s "
          f"({len(disc.store.excluded)} excluded)")

    # 2. pystreme
    t0 = time.perf_counter()
    motifs = disc.fit(n_motifs=args.n_motifs, kernel=args.kernel, verbose=True)
    t_pystreme = time.perf_counter() - t0
    pmeme = out / "pystreme.meme"
    disc.to_meme(motifs, pmeme)
    import pickle
    with open(out / "motifs.pkl", "wb") as fh:  # the Motif objects, for re-rendering figures without a rerun
        pickle.dump(motifs, fh)
    n_sites = disc.to_bed(motifs, out / "pystreme_sites.bed")
    print(f"wrote {n_sites} sites to {out / 'pystreme_sites.bed'}")
    print(f"pystreme.fit: {len(motifs)} motifs in {t_pystreme:.1f}s on {args.device}")
    for m in motifs:
        print(f"  {m!r}")

    lines = [f"# L6/L8: pystreme vs STREME on {bed.name} ({disc.store.n_seqs} peaks x {disc.store.length} bp)", ""]
    lines += [f"- device: {args.device}, kernel: {args.kernel}, widths {args.minw}-{args.maxw}, n_motifs {args.n_motifs}",
              f"- extraction: {t_extract:.1f}s", f"- **pystreme.fit: {t_pystreme:.1f}s** (in-process, after extraction)"]

    ann = None
    if args.annotate:
        from pystreme.results import annotate, summary
        summary(motifs).to_csv(out / "summary.csv", index=False)
        ann = annotate(motifs, args.annotate, top=3)
        ann.to_csv(out / "annotation.csv", index=False)
        lines += ["", f"## pystreme motifs annotated against `{Path(args.annotate).name}` (in-process Pearson ranking, top 3)", "",
                  "| motif | n_sites | hold-out log p | central log p | best matches |", "|---|---|---|---|---|"]
        for m in motifs:
            hits = ann[ann["query"] == m.name].sort_values("rank")
            best = ", ".join(f"{h.target.split('.')[0]} ({h.mean_r:.2f})" for h in hits.itertuples())
            lines.append(f"| {m.name} | {m.n_sites} | {m.holdout_logp:.1f} | {m.central_logp:.1f} | {best} |")
    if args.report:
        import matplotlib
        matplotlib.use("Agg")
        from pystreme import plot
        # most significant first, database match in the title when --annotate was given
        plot.report(motifs, annotation=ann).savefig(out / "report.png", dpi=110)
        lines.append(f"- report figure: `{out / 'report.png'}` (ordered by hold-out p)")

    # 3. STREME on the identical FASTA
    if not args.skip_streme:
        sdir = out / "streme"
        t_streme = _run(["streme", "--p", str(fasta), "--dna", "--nmotifs", str(args.n_motifs),
                         "--minw", str(args.minw), "--maxw", str(args.maxw), "--oc", str(sdir)], out / "streme.log")
        smeme = sdir / "streme.txt"
        print(f"streme: {t_streme:.1f}s (process wall-clock incl. startup)")
        lines.append(f"- **streme: {t_streme:.1f}s** (subprocess wall-clock, including startup)")
        lines.append(f"- speedup: {t_streme / t_pystreme:.1f}x")

    # 4. Tomtom
    jaspar = out / "jaspar.meme"
    with open(jaspar, "w") as fh:
        for fx in sorted(FIXTURES.glob("*.meme")):
            fh.write(fx.read_text() if fh.tell() == 0 else _strip_meme_header(fx.read_text()))

    def block(title, meme, tomtom_best_fn):
        lines.extend(["", f"## {title}", "", "| motif | w | nsites | E | IC (bits) | best Tomtom match | E-value |", "|---|---|---|---|---|---|---|"])
        best = tomtom_best_fn()
        for name, w, nsites, ev, ic in _meme_summary(meme):
            hit = best.get(name)
            match = f"{hit['Target_ID']} ({hit['Target_consensus']}, {hit['Orientation']})" if hit else "-"
            e = f"{float(hit['E-value']):.2g}" if hit else "-"
            lines.append(f"| {name} | {w} | {nsites} | {ev:.2g} | {ic:.1f} | {match} | {e} |")

    if have_tomtom:
        block("pystreme motifs vs JASPAR SP1/KLF4", pmeme, lambda: _tomtom(pmeme, jaspar, out / "tomtom_py_jaspar.tsv"))
    else:
        lines += ["", "(tomtom not on PATH: JASPAR comparison skipped; load the MEME module for it)"]
    if not args.skip_streme:
        block("STREME motifs vs JASPAR SP1/KLF4", smeme, lambda: _tomtom(smeme, jaspar, out / "tomtom_streme_jaspar.tsv"))
        block("pystreme motifs vs STREME motifs", pmeme, lambda: _tomtom(pmeme, smeme, out / "tomtom_py_streme.tsv"))

    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary)
    print()
    print(summary)


def _strip_meme_header(text: str) -> str:
    """Keep only the MOTIF blocks of a MEME file (for concatenating fixtures)."""
    i = text.find("MOTIF")
    return text[i:] if i >= 0 else ""


if __name__ == "__main__":
    sys.exit(main())
