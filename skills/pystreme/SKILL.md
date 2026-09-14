---
name: pystreme
description: >-
  Use for any task involving pystreme — the lab's GPU-batched, in-process
  reimplementation of STREME for de novo motif discovery from peak sets
  (ChIP-seq, ChEC-seq, ATAC), plus its scanner, enrichment test, motif
  annotation, site tables/BED export and plots. Triggers on "find motifs
  in these peaks", "run STREME/HOMER-style discovery from Python", "scan a
  JASPAR/HOCOMOCO PWM over peaks", "is motif X enriched", "central
  enrichment", "export motif sites as BED", "name this motif", or anything
  touching MotifDiscovery/Motif. This is a router skill — read the relevant
  file under references/ before writing pystreme code, and read
  references/cluster.md before running anything on a cluster.
---

# pystreme

`pystreme` (package `pystreme`, repo
<https://github.com/vmindel/pystreme>) finds motifs de novo the way STREME does — seeds,
PWM refinement, hold-out significance, erasing, one motif per round — but
batched on the GPU and called from Python, returning `Motif` objects
instead of an HTML report. Validated head-to-head against STREME 5.5.0
(same primary motif, 8 of 10 motifs shared on 7232 SP1 ChEC-seq peaks; 62 s
on a reserved A40 vs 191 s, re-timed 2026-09-10) and against FIMO (scanner r = 1.00000). Full record
in `BENCHMARKS.md`; site docs in `docs/` (`mkdocs serve`).

## The whole API in one block

```python
from pystreme import MotifDiscovery, summary, annotate, plot

disc = MotifDiscovery.from_bed("peaks.bed", "genome.fa", size=200, device="cuda")
motifs = disc.fit(n_motifs=10)            # list[Motif], discovery order (only roughly by p), hold-out significant

summary(motifs)                            # DataFrame: name, width, n_sites, train/holdout/central log p ...
ann = annotate(motifs, "H12CORE_meme_format.meme", top=3)   # name them (HOCOMOCO v12, in-process Tomtom-style)
motifs[0].sites_frame()                    # per-peak best site, genome coordinates, strand, passing
disc.to_bed(motifs, "sites.bed")           # BED6 of every passing site
disc.to_meme(motifs, "motifs.meme")        # for Tomtom / FIMO
plot.report(motifs, annotation=ann)        # logo + positional histogram per motif, most significant first

hit = disc.scan(pwm)                       # best score/position/strand per peak for a known PWM
enr = disc.enrichment(pwm)                 # SEA-style enrichment of a known PWM vs matched control
```

Two numbers per motif, deliberately: `holdout_logp` is the one to quote
(threshold re-tested on held-out peaks); `log_pvalue` is the training-set,
selected, optimistic one. `central_logp` separates directly bound motifs
(sharply central) from co-factors. Both are natural logs.

## Task → reference file

| If the task is… | Read |
|---|---|
| run discovery on a peak set, choose `fit` arguments, controls, widths | [references/usage.md](references/usage.md) |
| interpret a result: which p-value, central enrichment, motif width, what "8/10 shared with STREME" means | [references/interpreting-results.md](references/interpreting-results.md) |
| running on a shared cluster (LSF/SLURM jobs, GPU queues, MEME module quirks) | [references/cluster.md](references/cluster.md) |
| exact signatures of `MotifDiscovery`, `Motif`, `results`, `plot`, `scan`, controls | [references/api.md](references/api.md) |
| reproduce or extend the validation (STREME/FIMO head-to-head, benchmarks, tests) | [references/validation.md](references/validation.md) |
| something looks wrong (slow, odd width, no motifs, warnings) | [references/pitfalls.md](references/pitfalls.md) |

## Non-negotiables

1. **GPU for `fit`.** CPU is ~7x slower on the same code (51 s vs 9 s for
   2000 peaks); fine for tests, not for analyses.
2. **On a shared cluster, go through the scheduler**, never a login node,
   and request memory explicitly (importing torch alone needs a few GB).
   See references/cluster.md.
3. **Quote `holdout_logp`, not `log_pvalue`.** And do not re-run `fit`
   over overlapping subsets and pick the best: that undoes the hold-out
   guarantee (vary `seed`, keep an untouched set).
4. **Peaks must be centered** (summits, or fixed-width around summits).
   Broad peaks trigger a warning and make the positional statistics
   meaningless.
