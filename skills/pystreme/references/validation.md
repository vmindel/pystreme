# Validation and benchmarks — what is proven, how to re-run it

Record: `BENCHMARKS.md` (also rendered in the docs site). Layers follow
`DESIGNDOC.md` §4.

| layer | what | status |
|---|---|---|
| L1 scanner vs FIMO | 2000 SP1 peaks, JASPAR SP1/KLF4, order-0 bg | **pass**: r = 1.00000, 99.95-100% identical/tied argmax. Needs complement-averaged background on both sides (MEME tools average A/T and C/G when scanning both strands) |
| L2 statistics vs brute force | `optimal_threshold` vs `scipy.stats.fisher_exact` loop; bounded search vs all-exact | tests |
| L3 k-mer counting | vs `collections.Counter`, N/edge exclusion | tests |
| L4 controls | dinucleotide preservation, N fixation, GC histogram matching | tests, both shuffle engines |
| L5 synthetic recovery | implanted motifs recovered by `discover`/`fit` | tests (`test_discovery.py`) |
| L6 vs STREME | 7232 SP1 peaks: same primary, 8/10 shared (Tomtom E < 0.01) | **pass**; yeast ChEC-seq regime *not* done |
| L7 determinism | same seed same output | tests |
| L8 wall-clock | 7232 peaks x 10 motifs: 61.7 s (A40, reserved with `j_exclusive=yes`, 2026-09-10) vs STREME 191 s; 2000 peaks: 9.3 s (H100, pre-review) vs 14.2 s | recorded |

## Re-running

```bash
# STREME head-to-head + Tomtom + figure + HOCOMOCO annotation + sites BED (GPU job, MEME module)
python scripts/validate_streme.py --peaks peaks.bed --genome genome.fa --device cuda \
       --n-motifs 10 --report --annotate H12CORE_meme_format.meme --out validation/run      # --n-peaks N to subset, --skip-streme

# FIMO agreement (CPU job, MEME module)
PYTHONPATH=scripts python scripts/validate_fimo.py --peaks peaks.bed --genome genome.fa --n-peaks 2000 --out validation/l1

# scan kernel micro-benchmark (GPU job)
python scripts/bench_scan.py

# tests
python -m pytest -q            # CPU, ~30-90 s; 2 GPU tests skip without CUDA
```

On a cluster, through the scheduler (references/cluster.md). Outputs: `<out>/summary.md`,
`pystreme.meme`, `pystreme_sites.bed`, `motifs.pkl` (the Motif objects:
re-render figures on CPU without a rerun), `report.png`, `summary.csv`,
`annotation.csv`, `streme/`, `tomtom_*.tsv`. With `--skip-streme` the
MEME module is optional (the JASPAR Tomtom block is then skipped).

## Adding a validation on a new dataset

1. `from_bed` with the dataset's peaks (summit-centered) and genome; note
   `disc.store.excluded`.
2. `fit(n_motifs=10, verbose=True)` on GPU; `summary`, `annotate`, `to_bed`.
3. `validate_streme.py --peaks ... --genome ...` for the STREME/Tomtom
   comparison; record in `BENCHMARKS.md` with device, peak count, timings,
   shared motifs.
4. The yeast ChEC-seq regime (different composition, sharper centering) is
   the one the design doc wanted and has not been run yet.

## Speed history (why things are the way they are)

1673 s → 62 s came almost entirely from statistics, not scanning:
`scipy.stats.hypergeom.logsf` is a per-element Python loop, and even the
C `sf` is O(draws) per element. Fixes live in `statistics.hypergeom_logsf`
(vectorized + log-space series fallback), `seeds.top_seeds` (binomial
pre-ranking, exact tail for the top 2000 words) and
`statistics._fisher_logsf_min_candidates` (O(1) bounds per cut, exact
tail only for cuts that can win). Do not reintroduce scipy tails in a
per-candidate loop.

2026-09-10: the review's exact-argmin fix made `_fisher_logsf_min_candidates`
resolve every surviving cut even when none was significant -- ~11k exact
tails per call, once per seed per refinement iteration -- so a fit that
took 85 s took 413 s, with an identical motif table. Fixed by a Bonferroni
noise floor (stop once a truncated batch finds nothing significant) and a
`1/|support|` lower bound for cuts at or below the mode; now 61.7 s. Time
on a reserved card (`-gpu "num=1:j_exclusive=yes"`): shared-card timings
were off by up to 2x, in both directions, and hid this regression.
