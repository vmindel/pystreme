# Interpreting results

## A `Motif`

```
Motif(1-GCCCCGCCCC, width=10, n_sites=5716, train_logp=-2485.5, holdout_logp=-280.5, central_logp=-370.7)
```

| field | meaning |
|---|---|
| `name` | `<round>-<consensus>`; consensus uses N where no base reaches 40% |
| `pwm` | (4, w) probabilities, rows A C G T |
| `threshold` | log-odds cut (nats) chosen on the training set |
| `log_pvalue` | Fisher's exact log p on the *training* set at that cut (any primary/control sizes) — selected, optimistic. Ranking only |
| `holdout_logp` | same cut re-tested on held-out peaks — **the significance to quote**; `None` when there was no hold-out |
| `holdout` | the full `ThresholdResult` (2x2 table) behind it |
| `sites` | `ScanResult` for every primary peak (best score/position/strand; `-inf` = no valid window) |
| `n_sites` | primary peaks whose best site clears `threshold` |
| `positions` | passing-site centers relative to the peak center, bp |
| `central_logp` / `central` | CentriMo-style central enrichment: binomial test of sites within ±h of center vs uniform, best h, Bonferroni over h. One-sided: depletion at the center shows as 0.0 |
| `intervals` | the extracted windows `sites` is indexed by (for `sites_frame`) |

`exp(holdout_logp)` is the p-value. It is a raw hold-out p-value for this
one motif; STREME reports the same kind of number. It is not corrected
for the number of motifs found.

## Reading a list of motifs

`fit` returns discovery order (one motif per round), which is only
roughly monotone in significance: each round is the best of what is
left after erasing, and the hold-out re-test is noisy (on the 10-motif
SP1 run rounds 3/4 and 7/8/9 are out of order). `plot.report` sorts by
hold-out p by default; `sort_by_pvalue(motifs)` / `summary(motifs,
order="pvalue")` do it for lists and tables. Names keep the round prefix.


On SP1 ChEC-seq peaks (7232, 10 motifs) the pattern to expect from any
sequence-specific factor:

- **Round 1**: the targeted factor's own motif, thousands of sites, huge
  hold-out significance, very central (log p -371).
- **Next rounds**: co-factors and co-occurring elements (NF-Y, NRF1, AP-1,
  ATF, ETS...), hundreds to ~2000 sites, hold-out log p -10 to -80,
  central enrichment weaker or absent.
- **Variants of round 1** can reappear later (round 6 `CCCGCCC`, a KLF/SP
  variant with 2449 sites): erasing removes the best site per peak per
  pass, and a family with many near-sites leaves residual signal. These
  are real, not artefacts; `annotate` names them as the same family.
- **Tail** (hold-out log p above about -10): repeats (GA-rich), A-rich
  FOX-like sites, rare long motifs. Significant, but treat as leads.

`central_logp` orders the list by plausibility of direct binding better
than the enrichment p-value does. A strongly enriched but non-central
motif is a co-occurring element or a poorly centered peak set.

## Motif width

The enrichment objective saturates: once a motif separates peaks from
control as well as the data allow, widths within a couple of nats are
noise. `fit` trims flank columns below 0.3 bits, extends while the next
column has ≥ 0.3 bits, and among widths within `width_tolerance` nats
prefers the most informative motif. Reported widths are the informative
core; STREME reports the same motif with degenerate flanks
(`HRGCCCCGCCCCYN` vs `GCCCCGCCCCC`). Neither is "more right".

## Comparing with STREME / a database

- `annotate` gives Tomtom's Pearson score (sum of per-column r over the
  best overlap ≥ 5 columns) and `mean_r`; mean_r ≥ 0.9 is a confident
  family assignment; 0.7-0.85 is "related". No p-value: for a claim, run
  real Tomtom on `to_meme` output (see references/validation.md).
- "8 of 10 shared with STREME" (BENCHMARKS.md) means Tomtom E < 0.01
  between the two tools' motifs; the unshared ones on each side were the
  weakest in both lists. Expect this level of agreement, not identity:
  different controls (shuffles are random), different width conventions.

## Determinism

Same `seed` → same control set, same split, same motifs, on the same
device. CPU vs GPU, or conv vs gather, can differ in flank columns and
site counts by a few percent (fp16 storage in conv; tie-breaks). Not in
which motifs are found.
