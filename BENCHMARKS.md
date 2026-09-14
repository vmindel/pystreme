# Validation and benchmark record

Results of `scripts/validate_streme.py` (DESIGNDOC.md §4 L6 and L8), kept
here rather than in scrollback. Every run below is on the first 2000 peaks
of the SP1 ChEC-seq set (`SP1FULL_peaks.bed`, mm10, 200 bp windows, widths
6-15, `n_motifs=3`, dinucleotide-shuffle control on both sides), run through
`bsub` on WEXAC. STREME is MEME 5.5.0, given the *identical* FASTA that
pystreme scanned, building its own shuffled control as pystreme does.

## L1 -- scanner vs FIMO (2026-09-07)

`scripts/validate_fimo.py`, 2000 SP1 peaks x 200 bp, JASPAR SP1 (MA0079.1)
and KLF4 (MA0039.1) smoothed with a 0.01 pseudocount and handed to both
tools, one order-0 background (complement-averaged, as MEME tools do when
scanning both strands), FIMO `--text --thresh 1.0 --motif-pseudo 0`:

| motif | kernel | Pearson r (best score per peak) | identical argmax | identical or tied within FIMO's 0.01-bit resolution |
|---|---|---|---|---|
| MA0039.1 KLF4 | gather / conv | 1.00000 | 99.45% | 100.00% |
| MA0079.1 SP1 | gather / conv | 1.00000 | 98.75% | 99.95% |

**Pass** (r > 0.999; >= 99% identical-or-tied). The remaining
disagreements are peaks whose top two windows differ by less than
FIMO's output resolution. Before the background was complement-averaged
the argmax agreement was 98.3% with r = 0.9997 -- the difference between
the two tools' backgrounds, not the scanners. FIMO scores with order-0
frequencies whatever the background file's order, so the order-2 cumsum
term (§2.3) is validated by the naive-reference tests in
`tests/test_sequence_store.py`, not by FIMO.

## Scan kernels (§3.1) -- `scripts/bench_scan.py` (2026-09-07)

40 PWMs (widths 6-15) x 7232 x 200 bp, both strands, mean of 5 runs:

| device | gather (fp32) | conv (one-hot; fp16 on CUDA) |
|---|---|---|
| CPU (one `short` slot) | 6953 ms | 980 ms |
| NVIDIA L40S | 14 ms | 7 ms |

Hence the defaults: conv on CPU (7x, and fp32 there), gather on CUDA --
per scan conv is 2x faster at 40 PWMs, but `fit` scans a few PWMs at a
time and its wall-clock was the same with either kernel (9.3 s gather vs
10.2 s conv, 2000 peaks, H100), so the fp32 kernel wins on
determinism. Both CUDA kernels match the CPU fp32 reference
(`tests/test_scanner.py::test_cuda_kernels_match_cpu_reference`).

## Dinucleotide shuffle (2026-09-07)

`control.DinucShuffle`, 2000 x 200 bp: pure-Python reference 1.08 s,
numba engine (default when numba imports; same algorithm, same
property tests) 0.02 s. On the full 7232-peak run this was ~7 s of the
87 s.

## Extraction (2026-09-07)

`SequenceStore.from_bed` reads the genome in sorted, merged blocks
(windows on one chromosome closer than 1 Mb share one pyfaidx slice).
All 7232 SP1 peaks from mm10: 7.5 s on a `short` node (the per-peak
version took 69 s cold on a GPU node earlier that day, 0.4 s with the
genome already in the page cache -- I/O, not CPU). Output identical.

## L6 -- same motifs as STREME? (2026-09-07)

Yes, all three, in the same order, on both kernels and on CPU. Tomtom
(pearson, `-thresh 10`) pystreme -> STREME, GPU conv run:

| pystreme (round, consensus) | STREME | Tomtom E |
|---|---|---|
| 1 `GCCCCGCCCCC` (w=11, 1536 sites, hold-out log p -83) | 1 `HRGCCCCGCCCCYN` | 1.6e-13 |
| 2 `GCCAATCA` (w=8, 493 sites, hold-out log p -14) | 2 `RRCCAATCAGMR` | 3.1e-09 |
| 3 `GTGACTCA` (w=8, 373 sites, hold-out log p -11) | 3 `RTGASTCAY` | 4.7e-05 |

Biology: the SP1 GC-box (Tomtom to JASPAR MA0079.1 SP1, E = 8e-4, on both
tools), the NF-Y CCAAT box (SP1's classic co-factor), and AP-1. Every
pystreme motif is significant on the held-out 10% (`Motif.holdout_logp`),
and the GC-box is strongly centrally enriched (central log p -120), the
AP-1 site only mildly (-10) -- the §2.7 output distinguishing the bound
factor from co-occurring ones, as intended.

Run-to-run differences (CPU vs GPU, conv vs gather) are in the flank
columns of motif 1 (`GCCCCGCCCCC` vs `GGCCCCGCCCC`) and in site counts by
a few percent -- the width-settling tolerance (`_discover_on`) and fp16
storage in the conv kernel, not in which motifs are found.

The width-6 to width-15 STREME motif 1 is 14 wide with degenerate flanks
(`HRGCCCCGCCCCYN`); pystreme trims flanks below 0.3 bits, hence 11.

## L8 -- wall-clock (2026-09-07)

Same 2000 peaks, `fit(n_motifs=3)` in-process, timed after extraction.
STREME's number is the subprocess wall-clock including its startup.

| run | pystreme fit | STREME | ratio |
|---|---|---|---|
| CPU, gather kernel, before profiling (unbatched refinement, exact hypergeometric on every candidate seed word) | 1673 s | 17 s | 0.01x |
| H100, gather kernel, batched refinement + binomial pre-ranking of seed words | 90 s | 15 s | 0.17x |
| H100, conv kernel (fp16 one-hot), same | 74 s | -- | 0.2x |
| H100, gather kernel, + bounded Fisher threshold search | **9.3 s** | 14.2 s | **1.5x** |
| H100, conv kernel, same | 10.2 s | -- | 1.4x |
| CPU (one `short`-queue slot), gather kernel, same code | 331 s | -- | 0.04x |
| CPU, conv kernel (now the CPU default) + numba shuffle | 51 s | -- | 0.28x |

The CPU rows are the same code path; the gap to the GPU rows is the scans
themselves (the design is GPU-first, §"Device"; CPU exists so the
correctness suite runs anywhere, and at 51 s it is usable in a pinch).

### Full peak set: 7232 peaks, `n_motifs=5`, H100, gather kernel

| | pystreme | STREME |
|---|---|---|
| wall-clock | **86.9 s** (fit, in-process) | 132.0 s (subprocess) |
| 1 | `GCCCCGCCCC` (5870 sites, hold-out log p -266, central -371) | `RGGGGCGGGGCYDG` -- Tomtom E 3.9e-11 |
| 2 | `AGCCAATCA` (1748, -62, central -139) | `YRRCCAATCRGMRV` -- E 4.8e-11 |
| 3 | `GTGACTCA` (1195, -47, central -26) | `RTGASTCAY` -- E 2.9e-05 |
| 4 | `GCGCAGGCGCA` (745, -44, central -9) | = STREME's 5 `WSTGCGCABGCGCRS` -- E 9.6e-10 |
| 5 | `GTGACGTCA` (923, -31, central -16) -- CRE/ATF | STREME's 4 is `CACTTCCGGKT` (ETS), not in pystreme's top 5 |

Same primary motif, four of five shared, one differing secondary on each
side (both real TF families) -- §4 L6's "same primary motif, overlapping
secondaries", at 1.5x STREME's speed. Central enrichment orders them as
one would hope: the targeted factor's own site far more central than any
co-factor.

Extraction (pyfaidx, per interval) took 69 s on the GPU node for 7232
peaks vs 6.5 s for 2000 on a CPU node -- I/O bound and outside the
timings above, but user-visible; fixed by the block reads above.

### Full peak set, `n_motifs=10`, NVIDIA A40 (2026-09-07)

`scripts/validate_streme.py --n-motifs 10 --report --annotate H12CORE_meme_format.meme`

> **Superseded 2026-09-10.** This is the pre-review record, kept as it was
> measured. The 62.0 s was taken on a shared card; the same code on a
> reserved A40 takes 84.9 s. The current code fits in 61.7 s reserved, and
> its motifs differ from the table below. See "Post-review correctness
> fixes" below and `docs/runs/sp1_full10/`.

| | pystreme | STREME |
|---|---|---|
| wall-clock | **62.0 s** (fit, in-process) | 191.3 s (subprocess) |

| pystreme | sites | hold-out log p | central log p | HOCOMOCO (in-process, mean r) | STREME match (Tomtom E) |
|---|---|---|---|---|---|
| 1 `GCCCCGCCCC` | 5716 | -280.5 | -370.7 | SP1, SP3 (1.00) | 1 `RGGGGCGGGGCYDG` (1.5e-11) |
| 2 `GCCAATCA` | 1804 | -82.3 | -128.0 | NFYA/B/C (1.00) | 2 `YRRCCAATCRGMRV` (3.6e-09) |
| 3 `GCGCNTGCGCA` | 838 | -38.8 | -11.9 | NRF1 (0.95) | 6 `WSTGCGCABGCGCRS` (1.6e-09) |
| 4 `GTGACTCA` | 1265 | -48.5 | -26.4 | JUN, FOSL2, JUNB (0.99) | 3 `RTGASTCAY` (9.4e-05) |
| 5 `GTGACGTCA` | 970 | -26.8 | -16.4 | ATF6A, ATF1, JDP2 (0.92) | 7 `GTSACGTSAC` (8.9e-07) |
| 6 `CCCGCCC` | 2449 | -20.4 | -99.5 | KLF11, SP3, KLF9 (1.00) | 5 `CCCGCCCMC` (8.4e-04) |
| 7 `AAANAAACA` | 709 | -9.4 | 0.0 | FOXL1, FOXC2 (0.90) | -- |
| 8 `AGGAGGAGG` | 3418 | -14.7 | -18.2 | ZN263, MAZ (0.91) | -- |
| 9 `CACTTCCGG` | 973 | -19.9 | 0.0 | FLI1, ETV5, ERG (1.00) | 4 `CACTTCCGGKT` (1.4e-07) |
| 10 `GAACTACAANTCCCA` | 213 | -13.3 | 0.0 | ZN143, ZNF76 (0.99) | 8 `RACTACAAYTCCCAG` (4.2e-10) |

Eight of ten shared; pystreme's two extras (a FOX site and a GA-repeat /
ZNF263-MAZ site) and STREME's two (`TGATTGACA`, an octamer `ATTTGCATAN`)
are all at the weak end of both lists. Central enrichment again orders
the list by plausibility of direct binding: the GC-box and the second
KLF/SP variant far more central than any co-factor, the FOX, ETS and
ZNF143 sites not central at all. Report figure: `docs/img/report_sp1_10.png` (sorted by hold-out p, HOCOMOCO match per row);
the run's `summary.md`, `annotation.csv` and `pystreme.meme` are kept
under `docs/runs/sp1_full10/`.

Same three motifs in every row. What the profiles showed and what changed:

1. `scipy.stats.hypergeom.logsf` is a per-element Python loop -- 95% of the
   synthetic-test runtime. Replaced by `statistics.hypergeom_logsf`.
2. Even scipy's C `hypergeom.sf` costs O(draws) per element; called on
   every candidate seed word (1e5-1e6 words against a 4e5-window pool) it
   was 91% of the *real-data* runtime -- not scanning, which was 30 s for
   ~1000 scans on CPU. Now pre-ranked with the O(1) binomial tail, exact
   hypergeometric only for the top 2000 words per width.
3. Refinement batched per width (one scan per store per iteration for all
   seeds of that width, §2.1) instead of per seed.
4. The threshold search (§2.5) called the C `hypergeom.sf` on every
   candidate cut -- O(n_seqs) each, O(n_seqs^2) per search, 1700 searches
   per fit: 85% of the GPU wall-clock, against 6% for all the scans. Now
   every cut gets O(1) bounds on its Fisher tail and only cuts that can
   still be the optimum are evaluated exactly (`statistics.
   _fisher_logsf_min_candidates`; the argmin is provably exact).

The remaining ~9 s on 2000 peaks is dominated by per-iteration Python and
device-sync overhead in the refinement loop, not by GPU work (~3 s of
conv/gather time). The GPU's advantage should grow with input size; the
full 7232-peak comparison is below.

## Post-review correctness fixes (2026-09-09/10)

The scoring changes from the 2026-09-08 external review -- `fit` now
re-fits the reporting store's background on the same training control the
threshold was learned from (it kept the setup background, fitted on the
whole unerased control), background contexts containing an `N` or an
erased base back off one Markov order at a time instead of reading the `N`
as a `T`, and `extend_flanks` re-fits its threshold per width -- change
the numbers. Same 2000 peaks, same seed, CPU, `--skip-streme`
(`validation/sp1_2000_postreview/`):

| | before | after |
|---|---|---|
| 1 | `GCCCCGCCCCC` w11, 1534 sites, E 1.5e-37 | `GCCCCGCCCCC` w11, 1525 sites, E 2.7e-42 |
| 2 | `CTGATTGGC` w9, 480 sites, E 2.0e-07 | `AGCCAATCA` w9, 248 sites, E 1.7e-04 |
| 3 | `GTGACTCA` w8, 335 sites, E 4.9e-06 | `GTGACTCA` w8, 355 sites, E 2.4e-03 |

Same three motifs (SP1/KLF, the NF-Y CCAAT box -- reported on the other
strand -- and AP-1), so this is a change in the numbers attached to them,
not in what is found. Rounds 2 and 3 move most because they scan erased
stores, which is exactly where the fabricated `T` contexts were. On the
full 7232-peak set the graded back-off also recovers ZNF143
(`GAACTACAANTCCCA`, 214 sites, hold-out log p -14.0), which a flat
order-0 fallback had lost in the erased A-rich neighbourhoods.

L1 (scanner vs FIMO) is unchanged, byte for byte -- it is order-0, where
there is no context to fabricate.

### The 4.8x regression, and where the time went (2026-09-10)

Timed on one A40 reserved with `j_exclusive=yes` (earlier figures for
this section, 62 / 130 / 202 s, were taken on shared cards and are
withdrawn -- one of them was measured while another job held the same
GPU). Full 7232-peak SP1 set, 200 bp, widths 6-15,
`n_motifs=10`, fit only:

| code | fit | motifs |
|---|---|---|
| pre-review `b6b5f3c` | 84.9 s | 10 |
| review fixes, exact Fisher argmin (finding #6 as first implemented) | 412.8 s | 10 |
| + noise-floor bail-out | 286.4 s | 10, identical |
| **+ below-mode lower bound (current)** | **61.9 s** | **10, identical** |

Finding #6 turned `_fisher_logsf_min_candidates`'s `max_exact` from a
hard cap into a doubling batch so the argmin was always exact. That is
free when a cut is significant (the bounds prune to a handful) and
ruinous when none is (11.5k of 14.5k candidates survive, all resolved
exactly), and `refine` asks for a threshold once per seed per iteration.
The exact search cost 4.8x the whole fit and did not change one reported
motif. Two changes, both keeping every returned value a valid upper
bound on the true tail:

- **noise floor**: once a truncated batch fails to bring the ceiling
  under a Bonferroni bar over the candidates searched
  (`log(0.05 / n_candidates)`), stop -- the argmin is exact whenever any
  cut clears that bar, approximate only among cuts that don't;
- **below-mode bound**: the `pmf(k)` lower bound is tiny far down the
  left tail, so cuts at or below the mode -- no enrichment at all, tail
  near 1 -- survived every ceiling and were 99% of all exact evaluations.
  For `k <= mode` the tail contains `pmf(mode) >= 1/|support|`, which
  removes them for nothing.

The result is under the pre-review time because the old cap still
resolved 256 candidates per call; the median call now resolves 3. The
graded background and its fit-once refactor (one model per round applied
to five stores, 0.71 s -> 0.006 s per round) are inside these numbers.
