# Running discovery

## Inputs

`MotifDiscovery.from_bed(peaks, genome, size=200, bg_order=2, max_n_frac=0.2,
device="cuda", widths=range(6, 16), control="dinuc_shuffle", holdout_frac=0.1,
revcomp=True, seed=0)`

- `peaks`: BED path (columns beyond 3 ignored; 4th = name) or a list of
  `Interval`. `genome`: indexed FASTA (`.fai` alongside) or a `pyfaidx.Fasta`.
- Extraction is `size` bp centered on each peak's midpoint (HOMER `-size`
  semantics). Off-chromosome, unknown-chromosome and >`max_n_frac`-N peaks
  are dropped with one warning; `disc.store.excluded` lists them.
- Alternatives: `MotifDiscovery.from_fasta(path)` (equal-length sequences),
  `MotifDiscovery.from_sequences(list_of_strings)`.
- Genome reads are block-merged (fast); 7232 mm10 peaks extract in ~8 s.

## `fit`

`disc.fit(primary_idx=None, n_motifs=3, pvalue_thresh=0.05, *, widths=None,
control=None, holdout_frac=None, seed=None, n_per_width=4, n_iter=20,
revcomp=None, kernel="auto", n_per_seq=1, bg_order=None, patience=3,
keep_insignificant=False, min_holdout=10, min_flank_ic=0.3,
width_tolerance=2.0, verbose=False) -> list[Motif]`

Per call: build control → split train/hold-out → per round: seeds →
batched refinement → best motif on train → re-test its threshold on
hold-out → report sites/positions/central enrichment on *all* primary
peaks → erase its sites everywhere → next round. Stops after `n_motifs`
significant motifs, or `patience` consecutive insignificant ones, or when
no seed words remain.

| argument | use it when |
|---|---|
| `primary_idx` | discovery on a subset of `disc.store` (e.g. one cluster); the control is built to match that subset |
| `n_motifs`, `patience` | how many motifs you want; 10 is fine on thousands of peaks |
| `pvalue_thresh` | hold-out p-value cutoff (0.05). Raw, not multiple-testing corrected |
| `widths` | range to *search*; the reported width is settled per motif (trim/extend by information) |
| `control` | `"dinuc_shuffle"` (default), `control.DinucShuffle(kmer=3)`, `control.GCMatched(pool=SequenceStore)`, `control.Explicit(...)`, or a `SequenceStore` / `list[str]` / index array |
| `n_per_seq` | more shuffles per peak (2-3) tightens the control at proportional cost |
| `holdout_frac` | 0.1 default; 0 disables (then only the optimistic train p-value exists, and `holdout_logp` is `None`) |
| `seed` | reproducibility: same seed, same control, same split, same motifs |
| `n_per_width`, `n_iter` | cost knobs (seeds refined per width, iterations per seed) |
| `keep_insignificant` | also return rejected motifs (they are erased before continuing regardless) |
| `kernel` | leave `"auto"` (conv on CPU, fp32 gather on CUDA) |
| `verbose` | one line per round |

`disc.discover(widths=..., ...)` is one round without hold-out/erasing —
a quick "what's the top motif" with only a training p-value.

## Controls, in practice

- Dinucleotide shuffle is what STREME does by default; it asks "is this
  motif more frequent than its own composition predicts".
- `GCMatched(pool=...)` asks "more frequent than in other regions of the
  same GC": for GC-rich factors (SP/KLF) this can make the *true* motif
  look weak — that is information about composition, not a negative
  result (DESIGNDOC.md §2.6). The pool must be a `SequenceStore` of
  candidate regions at the same `size` (e.g. `SequenceStore.from_bed` on
  all accessible regions or another factor's peaks).
- `Explicit`: your own control set (a `SequenceStore`, strings, or an
  index array into the same store — e.g. peaks of a different condition).

## Known motifs

```python
from pystreme.meme_io import read_meme
pwm = read_meme("motifs.meme")["MA0079.1"].pwm        # (4, w) probabilities, rows ACGT
hit = disc.scan(pwm)                                   # ScanResult: .scores/.positions/.strands, one per peak
hits = disc.scan([pwm_a, pwm_b])                       # batched; motif axis kept
enr = disc.enrichment(pwm, control="dinuc_shuffle")   # ThresholdResult: .threshold .log_pvalue .table
```

Scores are log-odds in nats vs the order-`bg_order` background; `-inf`
means no valid window (all-N peak).

## Outputs

- `summary(motifs)` → DataFrame; `motif.sites_frame()` → per-peak table
  (chrom/start/end/name, score, site_start/site_end, strand, offset,
  passing); `disc.to_bed(motifs, path)` → BED6 of passing sites
  (`motif|peak` names, score in nats, strand); `disc.to_meme(motifs, path)`.
- `annotate(motifs, db, top=3)` → DataFrame of best database matches by
  Tomtom's Pearson metric (sum of per-column r over the best alignment; no
  p-value). HOCOMOCO v12 core (`H12CORE_meme_format.meme`, from the HOCOMOCO website); JASPAR
  fixtures in `tests/fixtures/`.
- `plot.report(motifs, annotation=ann)` / `plot.logo(m)` / `plot.positions(m)` (matplotlib;
  report is sorted most significant first and shows the `annotate` rank-1 match per motif;
  `fit` order is round order, only roughly monotone in p -- `sort_by_pvalue(motifs)` when it matters)
  Axes/Figure; needs logomaker + matplotlib).

## Minimal script for a bsub job

```python
import sys, torch
from pystreme import MotifDiscovery, summary, annotate
peaks, genome, out = sys.argv[1:4]
disc = MotifDiscovery.from_bed(peaks, genome, size=200, device="cuda", seed=0)
motifs = disc.fit(n_motifs=10, verbose=True)
summary(motifs).to_csv(f"{out}/summary.csv", index=False)
annotate(motifs, "H12CORE_meme_format.meme").to_csv(f"{out}/annotation.csv", index=False)
disc.to_bed(motifs, f"{out}/sites.bed"); disc.to_meme(motifs, f"{out}/motifs.meme")
```

The reference for all of this end to end: `scripts/validate_streme.py`
(`--report --annotate <db>` also writes the figure and tables) and
`notebooks/smoke_test.ipynb`.
