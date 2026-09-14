# Validation

pystreme reimplements STREME, so the question is whether it finds what
STREME finds, how fast, and whether its motif scanner scores sequences the
way the MEME Suite does. All three were measured on real data against the
real tools. The full development log, with how the numbers evolved, is on
the [development log](benchmarks.md) page.

## Does it find the same motifs as STREME?

**Data:** 7,232 SP1 ChEC-seq peaks from mouse (mm10), 200 bp around each
peak centre. Both tools got the identical sequences, searched motif widths
6–15 for ten motifs, and used their default dinucleotide-shuffled control.
pystreme motifs were matched to STREME's with Tomtom; a match means
E < 0.01.

**Result:** the same primary motif, and **8 of 10 motifs shared**.

| pystreme motif | sites | hold-out log p | closest known factor | STREME motif (Tomtom E) |
|---|---|---|---|---|
| `GGCCCCGCCCC` | 5,559 | −280.9 | SP1 | `RGGGGCGGGGCYDG` (4e−11) |
| `GATTGGC` | 2,049 | −64.0 | NF-Y | `YRRCCAATCRGMRV` (1.5e−6) |
| `GTGACTCA` | 1,254 | −47.3 | JUN (AP-1) | `RTGASTCAY` (8.6e−5) |
| `GCGCNTGCGCA` | 712 | −34.7 | NRF1 | `WSTGCGCABGCGCRS` (4e−10) |
| `GTGACGTCA` | 930 | −28.5 | ATF / CREB | `GTSACGTSAC` (2.8e−7) |
| `CCCGCCC` | 3,622 | −23.4 | KLF / SP3 | `CCCGCCCMC` (8.4e−4) |
| `CACTTCCGG` | 1,263 | −26.0 | ETS | `CACTTCCGGKT` (1.3e−7) |
| `GAACTACAANTCCCA` | 214 | −14.0 | ZNF143 | `RACTACAAYTCCCAG` (7.5e−10) |
| `CTCCTCCTCC` | 1,894 | −8.9 | ZNF436 (weak) | none |
| `AAAAAAAAAAAAAAA` | 564 | −3.5 | ZNF362 (weak) | none |

The two pystreme motifs without a STREME match are low-complexity repeats
at the weak end of the list. STREME's two unmatched motifs, `TGATTGACA` and
`ATTTGCATAN`, are likewise its weakest. "Closest known factor" is the best
HOCOMOCO v12 match by `annotate`. Log p values are natural logs.

## How fast is it?

| | wall-clock |
|---|---|
| pystreme, one NVIDIA A40 GPU | **61.7 s** |
| STREME 5.5.0 (MEME Suite, CPU) | 191.3 s |

Same 7,232 peaks and settings as above. The pystreme time is the `fit` call
alone, on a GPU reserved for the job so no other work shared it. STREME's
is its process wall-clock. Run pystreme on a GPU: on a CPU it is several
times slower.

## Does the scanner score like FIMO?

**Data:** 2,000 of the same SP1 peaks, the JASPAR SP1 (MA0079.1) and KLF4
(MA0039.1) motifs, and the same order-0 background for both tools.

**Result:** each peak's best score correlates with FIMO's at
**Pearson r = 1.00000**, and the best site is the identical position and
strand, or an exact tie, in 99.95–100% of peaks. The remaining differences
come from FIMO rounding scores to two decimals.

The unit tests add more checks, mostly against brute-force reference
implementations: the k-mer counting, the background model, the controls,
the statistics and the refinement, plus recovery of motifs planted in
synthetic sequences.

## Reproduce it

Both scripts need the MEME Suite (`streme`, `tomtom`, `fimo`) on `PATH`,
a genome FASTA with its `.fai` index, and a summit-centred peak BED:

```bash
# pystreme vs STREME, with Tomtom, HOCOMOCO annotation and the report figure (GPU)
python scripts/validate_streme.py --peaks peaks.bed --genome genome.fa \
    --n-motifs 10 --device cuda --report \
    --annotate H12CORE_meme_format.meme --out validation/run

# scanner vs FIMO (CPU is fine)
PYTHONPATH=scripts python scripts/validate_fimo.py --peaks peaks.bed \
    --genome genome.fa --n-peaks 2000 --out validation/l1
```

## Limits of this validation

- **One factor, one dataset:** SP1 ChEC-seq peaks from mouse. Other
  factors, species and assays (ChIP-seq, ATAC, yeast ChEC-seq) have not
  been benchmarked.
- **One run each.** Timings vary about 1.5x between GPU models, and results
  can shift slightly with the random `seed` (the control shuffle and the
  hold-out split).
- **Agreement, not identity.** The goal is to find the motifs STREME
  finds, not to reproduce its output bit for bit; see the
  [design document](design.md).
