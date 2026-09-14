# pystreme — design document and implementation plan

A GPU-batched reimplementation of the STREME algorithm for de novo motif
discovery, callable in-process from Jupyter.

Status: design draft v2, pre-implementation. Open questions from v1 resolved in §8.

---

## 1. Goals and non-goals

### Goals

1. **In-process.** `motifs = disc.fit(idx)` from a notebook cell. No subprocess,
   no FASTA on disk, no HTML parsing.
2. **Fast enough to iterate.** Target: a 10k × 300 bp peak set, widths 6–15,
   3 motifs, in under 10 s on GPU. STREME on the same input is roughly
   1–5 min plus process overhead.
3. **Reliable in the sense that matters:** recovers the motifs STREME recovers,
   with comparable significance ranking. Validated against real ChIP/ChEC data,
   not just synthetic.
4. **Composable.** The PWM scanner is exposed as a standalone primitive, because
   "score these regions with this motif" is a more common need than de novo
   discovery.

### Non-goals

- **Bit-identical reproduction of STREME output.** The paper defers the exact
  shuffle algorithm, hold-out sampling, RNG, and erasing semantics to
  Supplementary Material. Chasing identity would dominate the schedule and buy
  nothing. Agreement is measured by motif recovery, not by diff.
- Non-DNA alphabets. DNA only, with an option for a non-complementable mode.
- Variable-length input. See §2.2 — fixed width is a design commitment, not a
  limitation to be lifted later.
- MEME's HTML/report machinery. We emit MEME minimal motif format (so Tomtom,
  FIMO, and gimmemotifs can consume our output) and Python objects.
- Replacing the MEME suite. This is a fast path for a common case.

---

## 2. Key design decisions

### 2.1 The suffix tree is replaced, not reimplemented

STREME's suffix tree serves two purposes: exact/approximate word counting during
seed evaluation, and best-site lookup during refinement. Refinement dominates the
runtime (NREF=4 seeds × ~10 widths × NITER=20 iterations ≈ 800 full-dataset
passes), and for that job the tree is a poor fit on modern hardware.

We replace it with:

- **Seed counting** → direct k-mer counting. 2-bit rolling encode into a
  `4^w` count array for w ≤ 12; sort-based counting above that. One pass per
  width, vectorized, trivially parallel.
- **Best-site lookup** → batched 1-D convolution. A PWM log-odds scan *is* a
  convolution over one-hot encoded sequence. This is the central insight the
  whole design rests on.

No suffix tree or suffix array anywhere in the implementation.

### 2.2 Fixed-width windows, HOMER `-size` semantics

All sequences are extracted as `center ± flank`, so every sequence has identical
length `L = 2·flank`. This is a commitment, not a default:

```python
disc = MotifDiscovery.from_bed(peaks, genome, size=200)   # ±100 from center
```

Three consequences worth stating explicitly:

1. **The ragged code path does not exist.** The universe is one dense
   `(N, 4, L)` tensor. Per-sequence maxima are a `max` over the last axis, not a
   segmented reduction. This removes the largest single source of indexing bugs
   in the design.
2. **Fisher's exact test is effectively the only test.** STREME picks Fisher when
   mean primary and control lengths agree within 0.01%, else Binomial. With fixed
   width and shuffled or GC-matched controls, lengths agree exactly, always. The
   Binomial branch is implemented for correctness with user-supplied controls of
   differing length, but it is a cold path. (Correction, 2026-09-08: the first
   implementation switched on differing set *sizes* instead, which is wrong --
   Fisher's exact test is valid for any two counts -- and degenerate when no
   control sequence passes a cut; `optimal_threshold` now uses Fisher always.)
3. **Positional analysis becomes free.** The scanner already returns best-site
   positions, and every sequence is centered on the peak summit, so the
   positional distribution of sites requires no extra computation. See §2.7.

**Edge handling** must be decided explicitly rather than emerging from ragged
logic. Policy: sequences whose window would run off a chromosome end are dropped
with a warning and a returned index of exclusions. Sequences containing `N` keep
their length; `N` positions are marked in the code array as the separator symbol
(4) and any window spanning one is masked out of the scan. Sequences exceeding a
configurable `max_n_frac` (default 0.2) are dropped.

### 2.3 Higher-order background costs nothing

STREME scores sites by likelihood ratio against a Markov background of order
`k`. Naively this makes the score context-dependent and breaks the convolution
formulation. It doesn't, if you split the terms:

```
score(window at i) = Σ_j log P_motif(x_{i+j} | j)   ← conv with the PWM kernel
                   − Σ_j log P_bg(x_{i+j} | x_{i+j-k..i+j-1})
```

The second term does not depend on the motif. So we precompute, once per
dataset, a per-position array `bgll[n, i] = log P_bg(x_i | context)` by a gather
into a `4^(k+1)` table, then take its cumulative sum along the sequence. The
background term for any window of any width is then a single subtraction of two
cumsum entries.

Consequence: arbitrary-order background is free at scan time, and changing the
background order does not require rescanning. Order 2 is the default.

**The model is fit on the control set, not the primary set.** This matters most
when the control is real genomic sequence (§2.6) — the background should describe
what the primary is being contrasted against.

**Contexts that cannot be read back off, one order at a time.** A context that
contains an `N` or an erased base has no valid order-`k` entry; the base is then
scored under the longest context that is still clean (order `k-1`, then `k-2`,
... down to the order-0 marginal), not read as if the `N` were a `T` and not
dropped straight to order 0. Dropping straight to order 0 flattens the background
around erased sites in low-complexity regions, and on the full SP1 set that was
enough to lose ZNF143.

**One model per round, applied to every store.** `background_model()` estimates
the tables once; `apply_background()` scores any store under them. Each round
of `fit` estimates the model on its training control and applies it to all five
stores (training/hold-out primary and control, plus the full primary used for
reporting), so every p-value in a round is computed against the same background.

### 2.4 Reverse complement is a second kernel, not a second pass

For a complementable alphabet, stack `[pwm, revcomp(pwm)]` into the kernel batch
and take an elementwise max over the two output channels. Same wall-clock cost as
one pass on GPU.

### 2.5 Threshold optimization is vectorized, or it is the bottleneck

Every refinement iteration sorts primary+control best-site scores and finds the
score threshold optimizing the enrichment p-value. Calling
`scipy.stats.fisher_exact` per candidate threshold would make this the dominant
cost — it is the single most likely way for a naive rewrite to end up slower
than the C implementation.

Instead: sort once, compute cumulative primary/control counts above each cut
point, then evaluate the test for *all* cut points in one vectorized call
(`hypergeom.sf` / `binom.sf` are vectorized over their parameters in scipy).
One call per iteration, not one per threshold.

In practice even one exact tail per cut point is too many: `hypergeom.sf` costs
~36 µs per candidate, there are ~2n candidates, and `refine` asks for a threshold
once per seed per iteration. So each cut first gets two O(1) bounds on its tail
(`pmf(k)` below, a geometric series above; cuts at or below the mode also get
`1/|support|` below, which rules them out), and only cuts whose lower bound beats
the best upper bound are evaluated exactly. When no cut is significant the bounds
cannot discriminate, so the search stops once a batch fails to beat a Bonferroni
noise floor, `log(0.05 / n_candidates)`. The result: the argmin is exact whenever
any cut is significant, every returned value is a valid upper bound (never more
significant than the truth), and the search costs a handful of exact tails per
call instead of thousands. Insisting on an exact argmin over noise cost 4.8x the
whole fit and changed no reported motif (BENCHMARKS.md, 2026-09-10).

### 2.6 Three control modes behind one interface

```python
control="dinuc_shuffle"        # default
control="gc_matched"           # requires a background region pool
control=my_sequences           # explicit list[str] or index array
```

**Dinucleotide shuffle.** Per-sequence Euler-path shuffle (Altschul–Erikson)
preserving dinucleotide frequencies and the positions of separator characters.
Built once at setup, so a plain numba/numpy loop is adequate — this is not a hot
path. `kmer` order configurable, default 2.

**GC-matched.** The user supplies a pool of candidate background regions (BED or
FASTA — typically random genomic windows of the same size, blacklist-filtered and
excluding the peaks themselves). We bin the primary set by GC content and sample
the pool to match the primary GC histogram, optionally also matching repeat
fraction. Sampling is without replacement where the pool allows.

**A caveat that matters for GC-rich factors.** GC-matching removes exactly the
compositional signal that makes a GC-box findable. For KLF/SP-family data,
expect the motif to weaken substantially or drop out relative to dinucleotide
shuffle. This is the control doing its job, and the *comparison between the two
controls* is more informative than either alone. Recommended practice: run both
and report both. The API makes this a one-line change by design.

### 2.7 Positional distribution as a first-class output

Because sequences are peak-centered and fixed-width, and the scanner already
returns argmax positions, we get a site-position histogram per motif for free.
Central enrichment distinguishes directly-bound motifs from co-occurring or
tethered ones, and is particularly informative for ChEC-seq, where signal is
sharply centered.

Exposed as `motif.positions` (per-sequence best-site offset relative to center)
plus a central-enrichment statistic in the spirit of CentriMo. Costs nothing;
included from Phase 4.

---

## 3. Architecture

```
SequenceStore          ← built once, GPU-resident
  ├── codes      (N, L)     uint8, 0-3 + 4 for N/separator
  ├── onehot     (N, 4, L)  fp16, optional — see §3.1
  ├── mask       (N, L)     bool, valid positions
  ├── bg_cumsum  (N, L+1)   fp32, order-k background log-likelihood cumsum
  └── kmer_counts{w: array} optional precomputed universe-wide counts

Control
  ├── DinucShuffle(kmer=2, seed)
  ├── GCMatched(pool, n_bins=20, match_repeats=False)
  └── Explicit(sequences)

Scanner
  scan(pwms, idx) -> (n_motifs, n_seqs) best scores, positions, strands
  ├── conv1d over one-hot   [or gather-sum over codes — §3.1]
  ├── subtract background window sums from bg_cumsum
  ├── mask windows spanning N or running off the end
  └── max over positions, and over fwd/rc channels

Statistics
  optimal_threshold(scores_primary, scores_control) -> (thr, logp, table)

SeedFinder
  top_seeds(primary_idx, control_idx, widths) -> {w: [seed_words]}

Refiner
  refine(seed_pwms_of_one_width, primary_idx, control_idx) -> best PWM

Discovery
  fit(primary_idx, …) -> [Motif, …]
```

### 3.1 Two scan kernels, benchmarked not assumed

- **conv1d on one-hot fp16.** Uses cuDNN and tensor cores. Memory:
  `N·4·L·2` bytes — 400 MB for 100k × 500 bp.
- **gather-and-sum on uint8 codes.** `score[i] = Σ_j PWM[j, codes[i+j]]`, one
  embedding lookup per offset. 8× less memory, no multiply-by-zero waste, but
  gives up the tensor-core path.

Both are ~30 lines. Phase 1 implements and benchmarks both; the winner becomes
the default and the other stays as the memory-constrained fallback.

Accumulation is fp32 regardless of storage dtype — fp16 accumulation over long
sequences loses meaningful precision in log-odds sums.

### Public API sketch

```python
disc = MotifDiscovery.from_bed(
    peaks, genome,
    size=200,                   # ±100 from peak center, HOMER semantics
    widths=range(6, 16),
    control="dinuc_shuffle",
    bg_order=2,
    holdout_frac=0.10,
    revcomp=True,
    device="cuda",
    seed=0,
)

motifs = disc.fit(primary_idx=None, n_motifs=3, pvalue_thresh=0.05)

for m in motifs:
    m.pwm            # (4, w) probabilities
    m.consensus      # 'NGGGGCGGGGN'
    m.holdout_logp   # significance on held-out sequences
    m.sites          # per-sequence best score + position + strand
    m.positions      # offsets relative to peak center
    m.central_logp   # central enrichment

disc.scan(pwm, idx)              # standalone scanner
disc.to_meme(motifs, "out.meme") # for Tomtom / FIMO / gimmemotifs
```

---

## 4. Correctness and validation plan

**L1 — Scanner vs. an established scanner.**
JASPAR KLF4 (MA0039) and SP1 (MA0079) — both GC-rich, a useful stress case for
background handling. Scan a real peak set with both pystreme and FIMO/MOODS at
order-0. Require Pearson r > 0.999 on per-sequence best scores and identical
argmax on ≥99% of sequences. Repeat at order-2 to check the cumsum term. Run
both scan kernels (§3.1) against each other for exact agreement.

**L2 — Statistics vs. brute force.**
Small synthetic score vectors (n ≤ 200), vectorized optimizer against an explicit
loop calling `scipy.stats.fisher_exact` at every cut point. Exact agreement on
threshold and p-value to float tolerance. Degenerate cases: all-primary-above,
ties at the threshold, zero cells.

**L3 — k-mer counting and edges.**
Against `collections.Counter` over explicit substrings. Words spanning an `N` or
a window boundary must not be counted. Chromosome-end exclusion tested against a
hand-built case.

**L4 — Control construction.**
Dinucleotide shuffle: verify preserved dinucleotide frequencies per sequence and
destroyed higher-order structure. GC-matched: verify the sampled pool's GC
histogram matches the primary's within tolerance, and that no primary region
leaks into the control.

**L5 — Synthetic recovery.**
Implant a known PWM into shuffled background at 50/20/10/5% of sequences and
varying information content. Measure Tomtom similarity to the implanted motif and
site-level position recovery. Gives a sensitivity floor to quote.

**L6 — Real data, head-to-head with STREME.**
A public KLF/SP ChIP-seq peak set plus one yeast ChEC-seq set (different base
composition, sharper centering — a genuinely different regime). Compare top 3
motifs by Tomtom. Success = same primary motif, overlapping secondaries. Run
under both dinuc and GC-matched controls and record the difference (§2.6).

**L7 — Determinism.** Same seed, same output. CPU/GPU differences confined to
score ties.

**L8 — Benchmark.** Wall-clock against STREME including its process startup.
This is the number that justifies the project.

---

## 5. Known risks

**Hold-out reuse under repeated calls.** STREME's hold-out gives a valid p-value
for *one* run. Calling `fit` many times over overlapping sequence sets and
selecting on the results invalidates it. Mitigations: resample the hold-out split
per call from a per-call seed, and keep a set of sequences no call ever touches
for final validation. A usage hazard, not a bug — documented prominently.

**GC-matched control and GC-rich motifs.** See §2.6. The risk is misreading a
weakened motif as a negative result rather than as compositional information.

**Peak centering quality.** Fixed-width windows assume the center is meaningful.
For broad or poorly-summited peaks this assumption fails, and the free positional
analysis (§2.7) becomes misleading rather than informative. Flag when input peak
widths are large relative to `size`.

**fp16 precision.** Storage fp16, accumulation fp32. Verified in L1.

**Memory.** The `4^w` count arrays are the real constraint: w=12 is 16.7M entries
(67 MB int32), w=15 would be 1e9. Hence the sort-based fallback above w=12.

**Scope creep.** Protein alphabet, position-specific priors, variable width,
plotting — all deferred or excluded.

---

## 6. Implementation plan

Each phase ends with something usable and tested. If the project stops after any
phase, what exists is still worth having.

### Phase 1 — The scanner

- `SequenceStore`: extraction at fixed width, encoding, N/edge masking,
  background model, cumsum.
- Both scan kernels (§3.1), benchmarked.
- MEME motif format reader/writer.
- Validation L1, L3.

**Deliverable:** `disc.scan(jaspar_pwm, idx)` — a fast in-notebook PWM scanner.
Independently useful even if nothing else gets built.

### Phase 2 — Controls and statistics

- Three control modes.
- Vectorized threshold optimizer, Fisher primary and Binomial cold path.
- Validation L2, L4.

**Deliverable:** enrichment testing for known motifs — a `SEA` equivalent,
in-process, with three background models.

### Phase 3 — Seeds and refinement

- k-mer counting and seed ranking; approximate-match re-ranking via the scanner.
- Refinement loop, batched per width.
- Validation L5.

**Deliverable:** single-motif de novo discovery.

### Phase 4 — The round loop

- Hold-out split and significance; erasing; multi-motif loop and stopping rules.
- Positional distribution and central enrichment (§2.7).
- Validation L6, L7, L8.

**Deliverable:** feature-complete pystreme.

### Phase 5 — Ergonomics (optional)

- Caching of universe-wide k-mer counts.
- Logo plotting via `logomaker`; positional histogram plots.
- Convenience constructors and a summary DataFrame output.

---

## 7. Dependencies

`numpy`, `scipy`, `torch` (CUDA). Optional: `pyfaidx` or `pybedtools` for
extraction, `numba` for the shuffle, `logomaker` for plots, `tangermeme` for
cross-checking the scanner in L1.

No compiled extensions, no Rust, no build step.

---

## 8. Resolved decisions

1. **Width:** fixed, `center ± flank`, HOMER `-size` semantics. Ragged path
   removed from the design entirely. → §2.2
2. **Device:** CUDA-first via torch, fp16 storage / fp32 accumulation, two scan
   kernels benchmarked in Phase 1. → §3.1
3. **Controls:** dinucleotide shuffle, GC-matched, user-provided — one
   interface, chosen at construction. → §2.6
4. **Motif widths:** default 6–15. Wide composite or dimeric sites are not
   covered by this default and need it raised explicitly.
