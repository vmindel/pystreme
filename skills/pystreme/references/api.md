# API cheatsheet

Module layout (`src/pystreme/`): `sequence_store` (extraction, background),
`scanner` (batched PWM scan), `control` (shuffle / GC-matched / explicit),
`statistics` (Fisher/binomial tails, central enrichment), `seeds` (k-mer
seeds), `refine` (PWM refinement, width trim/extend), `discovery`
(`MotifDiscovery`, `Motif`), `results` (tables, annotation, BED), `plot`,
`meme_io`. Top-level exports: `MotifDiscovery, Motif, SequenceStore,
summary, sites_frame, sites_bed, annotate`.

## `pystreme.discovery`

```python
MotifDiscovery(store, widths=range(6,16), control="dinuc_shuffle", holdout_frac=0.1, revcomp=True, seed=0)
MotifDiscovery.from_bed(peaks, genome, size, bg_order=2, max_n_frac=0.2, device="cpu", **ctor_kwargs)
MotifDiscovery.from_fasta(path, bg_order=2, max_n_frac=0.2, device="cpu", **ctor_kwargs)
MotifDiscovery.from_sequences(seqs, bg_order=2, device="cpu", **ctor_kwargs)

disc.fit(primary_idx=None, n_motifs=3, pvalue_thresh=0.05, *, widths, control, holdout_frac, seed,
         n_per_width=4, n_iter=20, revcomp, kernel="auto", n_per_seq=1, bg_order, patience=3,
         keep_insignificant=False, min_holdout=10, min_flank_ic=0.3, width_tolerance=2.0, verbose=False) -> list[Motif]
disc.discover(widths, control, idx, n_per_width, n_iter, revcomp, kernel, n_per_seq, bg_order, seed, ...) -> Motif
disc.scan(pwm | [pwms], idx=None, revcomp=None, kernel="auto") -> ScanResult   # single pwm: squeezed (n_seqs,)
disc.enrichment(pwm, control=None, idx=None, revcomp=None, kernel="auto", n_per_seq=1, bg_order=None, seed=None) -> ThresholdResult
disc.to_meme(motifs | {name: pwm} | [MemeMotif], path, background=None)
disc.to_bed(motifs, path, passing_only=True) -> int
disc.store  # SequenceStore

Motif: pwm, consensus, width, seed, threshold, log_pvalue, sites: ScanResult, n_sites,
       holdout_logp, holdout: ThresholdResult|None, positions, central_logp, central: CentralEnrichment|None,
       round, intervals; .name property; .sites_frame()
```

## `pystreme.results`

```python
summary(motifs, order="round") -> DataFrame   # name, round, consensus, width, n_sites, train_logp, holdout_logp,
                                        # order="pvalue": most significant first
sort_by_pvalue(motifs) -> list[Motif]   # by holdout_logp (log_pvalue without hold-out); fit() returns round order
                                        # central_logp, central_half_width, threshold, seed
sites_frame(motif) -> DataFrame         # chrom,start,end,name, score, site_start, site_end, strand, offset, passing
sites_bed(motifs, path, passing_only=True, sort=True) -> int
annotate(motifs, database_path | {name: MemeMotif}, top=3, min_overlap=5) -> DataFrame
                                        # query, rank, target, score, mean_r, offset, orientation, overlap
compare_pwms(query, target, min_overlap=5) -> (score, offset, orientation, overlap)
```

## `pystreme.plot` (matplotlib, logomaker)

```python
logo(motif_or_pwm, ax=None, title=None) -> Axes
positions(motif, ax=None, bins=40, title=None, shade_below_logp=log(0.05)) -> Axes
report(motifs, figsize_per_row=(9.0, 2.2), annotation=None, order="pvalue") -> Figure
    # annotation: DataFrame from results.annotate (rank-1 hit in each title) or {motif name: label}; order: "pvalue" | "round"
```

## `pystreme.sequence_store`

```python
SequenceStore.from_bed(peaks, genome, size, max_n_frac=0.2, device="cpu")
SequenceStore.from_fasta(path, device="cpu", max_n_frac=0.2)
SequenceStore.from_sequences(list_of_str, device="cpu")
store.codes (N,L) uint8 0-3 + 4=N; store.mask (N,L) bool; store.intervals; store.excluded
store.fit_background(order=2, source=None, pseudocount=1.0)   # source=control store per §2.3
store.background_model(order=2, pseudocount=1.0) -> BackgroundModel   # estimate once, from this store
store.apply_background(model)   # score any store under an existing model (what fit's round loop does)
# contexts containing an N or an erased base back off one Markov order at a time
store.subset(idx); store.erase(rows, starts, width); store.to_strings(); store.to_fasta(path)
```

## `pystreme.control`

```python
DinucShuffle(kmer=2, seed=None, engine="auto")   # engine: "numba" (default if importable) | "python"
GCMatched(pool: SequenceStore, n_bins=20, match_repeats=False)   # match_repeats not implemented
Explicit(sequences)                              # SequenceStore | list[str] | index array
resolve_control(control, store, n_per_seq=1, seed=None) -> SequenceStore
```

## `pystreme.scanner`, `pystreme.statistics`

```python
scan(pwms: list[np.ndarray], store, idx=None, revcomp=True, kernel="auto") -> ScanResult(scores, positions, strands)  # (n_motifs, n_seqs)
optimal_threshold(scores_primary, scores_control) -> ThresholdResult(threshold, log_pvalue, primary_above, primary_below, control_above, control_below)
enrichment_at_threshold(scores_primary, scores_control, threshold) -> ThresholdResult
central_enrichment(offsets, n_windows) -> CentralEnrichment(log_pvalue, half_width, n_central, n_sites, expected) | None
hypergeom_logsf(k, M, n, N); binom_logsf(k, n, p)   # fast, vectorized, no underflow
```

## `pystreme.meme_io`

```python
read_meme(path) -> {name: MemeMotif(name, pwm, alt_name, nsites, evalue)}   # JASPAR, HOCOMOCO, STREME output all parse
write_meme({name: pwm} | [MemeMotif], path, background=None)
```
