# Using pystreme

De novo motif discovery from Python, in-process, on the GPU. This is the
practical guide; `DESIGNDOC.md` has the reasoning, `BENCHMARKS.md` the
validation against real STREME.

## Install and run

```bash
./scripts/setup_env.sh          # creates the `pystreme` conda env, editable install
conda activate pystreme
pytest                          # CPU-only correctness suite, ~30 s
```

`fit` needs a GPU to be fast (on a CPU the same code is several times
slower; the correctness suite runs anywhere). On a shared cluster that
means a job on a GPU node, never the login node; see
[Running on a cluster](cluster.md).

Memory: reserve at least 4 GB for anything that imports torch; 16 GB is
comfortable for tens of thousands of 200 bp peaks.

## The two-line version

```python
from pystreme import MotifDiscovery, summary

disc = MotifDiscovery.from_bed("peaks.bed", "genome.fa", size=200, device="cuda")
motifs = disc.fit(n_motifs=3)
summary(motifs)
```

`from_bed` takes a BED (any columns after the first three are ignored; the
4th becomes the peak name) and an indexed FASTA (`.fai` alongside), and
extracts `size` bp centered on each peak's midpoint -- HOMER `-size`
semantics. Peaks that would run off a chromosome end, name an unknown
chromosome, or are more than `max_n_frac` N are dropped with one warning;
`disc.store.excluded` lists them with reasons. If your peaks are broad
(median width more than twice `size`) you get a warning: fixed windows
assume the center is meaningful, so use summit-centered peaks where you
can. `from_fasta` (equal-length sequences in a FASTA) and `from_sequences`
(a list of strings) exist for sequences that did not come from a genome.

## What `fit` does and what its arguments mean

Each call: build a control set, split primary and control into training
and hold-out, then repeat -- find the best motif on the training set,
re-test it on the hold-out, report it, erase its sites, go again -- until
the stopping rule fires.

| argument | default | meaning |
|---|---|---|
| `primary_idx` | all | row indices into `disc.store` that count as "primary" (e.g. one cluster of peaks); the control is built to match that subset |
| `n_motifs` | 3 | stop after this many *significant* motifs |
| `pvalue_thresh` | 0.05 | a motif counts if its hold-out p-value is at most this |
| `patience` | 3 | also stop after this many consecutive insignificant motifs |
| `keep_insignificant` | False | return the rejected motifs too (they are erased before continuing either way, as in STREME) |
| `widths` | 6..15 | motif widths to seed; the final width is settled per motif (see below), so this is a range to search, not a menu |
| `control` | `"dinuc_shuffle"` | see "Controls" |
| `holdout_frac` | 0.1 | fraction of primary (and control) held out for significance; hold-out is skipped with a warning below `min_holdout=10` sequences per side |
| `seed` | 0 | seeds the control construction and the split; same seed, same result |
| `n_per_width` / `n_iter` | 4 / 20 | seeds refined per width, refinement iterations per seed -- the cost knobs; STREME's own defaults are similar |
| `bg_order` | from `from_bed` (2) | Markov order of the background model, fit on the control set |
| `kernel` | `"auto"` | scan kernel; `"gather"` (default on CUDA) or `"conv"` (fp16 one-hot) |
| `min_flank_ic` / `width_tolerance` | 0.3 / 2.0 | width settling: trim/extend flank columns below/above this many bits; treat widths within this many nats of the best p-value as tied and prefer the more informative motif |
| `verbose` | False | print one line per round |

The constructor (`MotifDiscovery(store, widths=..., control=...,
holdout_frac=..., revcomp=..., seed=...)`, and hence `from_bed(...,
widths=...)`) sets per-object defaults for the same names.

### Controls

- `"dinuc_shuffle"` (default): each primary sequence shuffled preserving
  its dinucleotide counts, N positions fixed. `control.DinucShuffle(kmer=3)`
  preserves trinucleotides instead; `n_per_seq=2` makes two shuffles per
  sequence. Compiled with numba when available (`engine="python"` forces
  the reference implementation).
- `control.GCMatched(pool=SequenceStore)`: sample from a pool of candidate
  regions (e.g. all accessible regions, or peaks of another factor) to
  match the primary set's GC histogram. Read `DESIGNDOC.md` §2.6 before
  using it on a GC-rich factor: it can make the true motif look weak, and
  that is information, not a negative result.
- `control.Explicit(...)`, or just a `SequenceStore`, a list of strings,
  or an index array into `disc.store`: your own control set.

### How a motif's width is decided

The enrichment objective saturates once a motif separates primary from
control as well as the data allows, so a motif one column shorter or
longer scores within noise of the same p-value. `fit` therefore trims
flank columns carrying less than `min_flank_ic` bits, grows the motif
outward while the next column carries at least that much, and among
widths whose p-values are within `width_tolerance` nats prefers the most
informative one. Reported widths are usually the informative core; real
STREME tends to report the same motif with a couple of degenerate flank
columns.

## Reading a `Motif`

```python
m = motifs[0]
m.consensus      # 'GCCCCGCCCC'  (N where no base reaches 40%)
m.pwm            # (4, w) probabilities, rows A C G T
m.holdout_logp   # natural-log p-value on the held-out sequences  <- quote this one
m.log_pvalue     # natural-log p-value on the training set: selected, optimistic
m.n_sites        # primary sequences whose best site clears m.threshold
m.sites          # best score / position / strand for every primary sequence
m.positions      # passing-site centers relative to the peak center (bp)
m.central_logp   # CentriMo-style central enrichment (Bonferroni over window sizes)
m.central        # ...with the winning half-width and counts
```

Two p-values on purpose. The threshold was chosen on the training
sequences to be as significant as possible, so `log_pvalue` overstates;
`holdout_logp` re-tests that fixed threshold on sequences the search never
saw and is what `fit` uses to accept a motif. It is `None` when there was
no hold-out (`discover()`, or too few sequences), and then `log_pvalue` is
all you have. Both are natural logs: `exp(m.holdout_logp)` is the p-value.

`central_logp` is the second thing to look at. The bound factor's own
motif is usually sharply central and co-factors much less so (on the SP1
ChEC-seq peaks in `BENCHMARKS.md`: GC-box central log p -381, AP-1 -26). A motif with strong enrichment and no
central signal is worth a second look -- a co-occurring element, or a
peak set whose centers are not summits.

Repeated `fit` calls over overlapping subsets, selecting on the results,
quietly undo the hold-out guarantee (`DESIGNDOC.md` §5). Vary `seed` per
call and keep an untouched set of peaks for a final check if that is the
workflow.

## Tables, names, figures, files

```python
from pystreme import summary, annotate, plot

summary(motifs)                       # one row per motif
motifs[0].sites_frame()               # one row per peak: chrom/start/end, best site coordinates, strand, passing
ann = annotate(motifs, "H12CORE_meme_format.meme", top=3)   # Tomtom-style Pearson ranking vs a MEME database
plot.report(motifs, annotation=ann)   # logo + positional histogram per motif, most significant first;
                                      # ann = annotate(...) puts the best database match in each title
plot.logo(motifs[0]); plot.positions(motifs[0])
disc.to_meme(motifs, "motifs.meme")   # MEME minimal format, for Tomtom/FIMO/gimmemotifs
disc.to_bed(motifs, "sites.bed")      # BED6 of every passing site: chrom start end motif|peak score strand
```

`fit` returns motifs in discovery order (one per round). That order is
only roughly monotone in significance: each round takes the best motif
of what is *left* after erasing the earlier sites, and the hold-out
re-test is noisy. `plot.report` therefore sorts by hold-out p by default
(`order="round"` for discovery order); `summary(motifs, order="pvalue")`
and `sort_by_pvalue(motifs)` do the same for tables and lists. Names keep
their round prefix, so `3-GCGCNTGCGCA` is still round 3 wherever it sits.

The BED has one line per (motif, peak): that motif's best site in that
peak, in genome coordinates, only where it clears the motif's threshold
(`passing_only=False` for every peak's best window). Load it in a genome
browser next to the peaks, or intersect it with other annotations.

`plot.report(motifs, annotation=ann)` on the first 2000 SP1 ChEC-seq peaks (the
run recorded in the benchmarks), most significant first, with the HOCOMOCO
match in each title: the bound factor's GC-box is sharply central, the NF-Y
co-factor less so, AP-1 barely.

![plot.report on SP1 peaks](img/report_sp1.png)

`annotate` ranks by the sum of per-column Pearson correlations over the
best alignment (Tomtom's Pearson metric) and reports offset and
orientation; it does not compute Tomtom's p-values. For a validation claim
run real Tomtom on the exported file (`scripts/validate_streme.py` shows
how).

## Known motifs: scan and enrichment

```python
from pystreme.meme_io import read_meme
sp1 = read_meme("tests/fixtures/MA0079.1_SP1.meme")["MA0079.1"]

hit = disc.scan(sp1.pwm)                # ScanResult: best score/position/strand per peak (both strands)
hit = disc.scan([pwm_a, pwm_b])         # batched, motif axis kept
enr = disc.enrichment(sp1.pwm)          # SEA-style: optimal threshold + Fisher p vs a matched control
enr.log_pvalue, enr.table               # (primary_above, primary_below, control_above, control_below)
one = disc.discover(widths=[8, 10])     # a single round of fit, no hold-out
```

Scores are log-odds in nats against the order-`bg_order` background;
`-inf` marks a sequence with no valid window (all-N).

## Reproducing the benchmark

```bash
# needs the MEME suite (streme, tomtom) on PATH
python scripts/validate_streme.py --peaks peaks.bed --genome genome.fa \
    --n-peaks 2000 --device cuda --out validation/run
```

writes `validation/run/summary.md` with pystreme's and STREME's motifs,
Tomtom cross-matches, and both wall-clocks (run it on a GPU node; the script's docstring has the
exact invocation).
