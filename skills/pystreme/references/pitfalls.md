# Pitfalls and diagnostics

| symptom | cause | do |
|---|---|---|
| `fit` takes minutes on thousands of peaks | running on CPU | `device="cuda"` in `from_bed`, GPU bsub job |
| job killed, exit 137 / `TERM_MEMLIMIT`, right after start | `short` queue default 1 GB; torch import alone exceeds it | `-R "rusage[mem=6000]" -M 6000` or more |
| `ModuleNotFoundError: numpy` inside a MEME-module job | module's Python shadows the env | absolute env interpreter + `PYTHONNOUSERSITE=1` |
| "No such file" for a script under `/tmp` in a job | `/tmp` is node-local | put it under the repo (`.scratch/`) |
| `RuntimeError: no bg_order available` | store built without `from_bed`/constructor background | `bg_order=` on the call, or `store.fit_background(order=2)` |
| warning "hold-out disabled" | fewer than `min_holdout` (10) peaks on a side | more peaks, or accept training-only p-values (`holdout_logp is None`) |
| warning "median input peak width ... more than twice size" | broad peaks | summit-center them first; positional statistics are otherwise meaningless |
| warning "dropped N/M peaks" | off-chromosome / unknown chrom / N-rich | check `disc.store.excluded`; chromosome naming (`chr1` vs `1`) is the usual cause of `unknown_chrom` |
| `fit` returns `[]` | nothing significant on hold-out within `patience` rounds | `keep_insignificant=True` to inspect; check the control choice; try `discover()` for the top training motif |
| top motif looks like a truncated core (`CCCGCCC`) or has flat flanks | width settling (`min_flank_ic`, `width_tolerance`) | it is the informative core; lower `min_flank_ic` to keep weaker flanks, or compare with `annotate` |
| the same family appears twice (GC-box in rounds 1 and 6) | erasing removes one site per peak per pass; large families leave residue | expected; `annotate` names both the same |
| `central_logp == 0.0` with sites clearly *avoiding* the center | the test is one-sided (enrichment) | that is a real signal of flanking placement; look at `plot.positions` |
| GC-matched control makes the SP/KLF motif disappear | composition: GC-box frequency tracks GC content | expected; see DESIGNDOC.md §2.6, report both controls |
| `annotate` top hit has `mean_r` < 0.7 | nothing similar in the database | treat as novel/repeat; check `plot.logo` for low-complexity |
| different motifs on CPU vs GPU, or conv vs gather | fp16 storage in conv, tie-breaks | expected in flank columns / a few percent of sites, not in which motifs |
| `mkdocs serve` fails with "mkdocstrings not installed" | docs env lacks the plugin | `pip install -r docs/requirements.txt` |
| `ValueError: control sequences are N bp but primary sequences are M bp` | explicit/GC control extracted at a different width | re-extract the control at the primary `size`; the per-sequence tests need the same number of window starts on both sides |
| `GCMatched` warns "the pool has no sequence at all in ... GC bin(s)" | the pool does not cover the primary's GC range, so those controls are drawn from the whole pool and are *not* matched | supply a pool covering that range; the warning reports the GC gap it actually produced (a separate warning, "pool exhausted", is the milder with-replacement case where the match still holds) |
| forward-only (`revcomp=False`) discovery finds weak, near-random motifs | fixed 2026-09-09: seeds were canonicalized against their reverse complement even when the scan was single-stranded | update; seeds now keep the observed orientation when `revcomp=False` |
| `fit` several times slower than `BENCHMARKS.md` says, CPU-bound while the GPU idles | the Fisher threshold search resolving thousands of candidates exactly per seed per iteration (fixed 2026-09-10: noise-floor bail-out + below-mode bound; an exact argmin over noise cost 4.8x and changed no motif) | update; if it recurs, profile `_fisher_logsf_min_candidates` -- survivors at or below the mode are the tell |

## Things the tool does not do

- No multiple-testing correction across motifs (raw hold-out p-values, like STREME).
- No protein alphabet, no variable-width windows, no position-specific priors.
- `GCMatched(match_repeats=True)` is not implemented.
- `annotate` gives no p-value; use real Tomtom for claims.
- Hold-out validity is per call: repeated `fit` over overlapping subsets
  with selection on the results is a usage hazard (DESIGNDOC.md §5).
