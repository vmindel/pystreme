# L6/L8: pystreme vs STREME on peaks.bed (7232 peaks x 200 bp)

- device: cuda, kernel: auto, widths 6-15, n_motifs 10
- extraction: 3.8s
- **pystreme.fit: 61.7s** (in-process, after extraction)

## pystreme motifs annotated against `H12CORE_meme_format.meme` (in-process Pearson ranking, top 3)

| motif | n_sites | hold-out log p | central log p | best matches |
|---|---|---|---|---|
| 1-GGCCCCGCCCC | 5559 | -280.9 | -380.7 | SP1 (1.00), SP3 (1.00), SP1 (1.00) |
| 2-GATTGGC | 2049 | -64.0 | -122.4 | NFYC (0.99), NFYB (0.99), NFYA (0.99) |
| 3-GTGACTCA | 1254 | -47.3 | -25.5 | JUN (0.99), JUNB (0.98), FOSB (0.98) |
| 4-GCGCNTGCGCA | 712 | -34.7 | -10.3 | NRF1 (0.95), ZBT14 (0.86), ZBT14 (0.74) |
| 5-GTGACGTCA | 930 | -28.5 | -16.2 | ATF6A (0.92), ATF1 (0.91), JDP2 (0.89) |
| 6-CCCGCCC | 3622 | -23.4 | -172.5 | KLF11 (1.00), SP3 (0.99), KLF9 (0.99) |
| 7-CTCCTCCTCC | 1894 | -8.9 | -12.0 | ZN436 (0.87), SP5 (0.85), ZN263 (0.85) |
| 8-AAAAAAAAAAAAAAA | 564 | -3.5 | 0.0 | ZN362 (0.95), ZN613 (0.91), CPEB1 (0.84) |
| 9-CACTTCCGG | 1263 | -26.0 | 0.0 | ETS2 (0.99), ETV6 (0.99), FLI1 (0.99) |
| 11-GAACTACAANTCCCA | 214 | -14.0 | 0.0 | ZN143 (0.99), ZNF76 (0.97), CREM (0.78) |
- **streme: 191.3s** (subprocess wall-clock, including startup; recorded 2026-09-07 -- STREME is unaffected by the review fixes and was not re-run)
- speedup: 3.1x

## pystreme motifs vs JASPAR SP1/KLF4

| motif | w | nsites | E | IC (bits) | best Tomtom match | E-value |
|---|---|---|---|---|---|---|
| 1-GGCCCCGCCCC | 11 | 5559 | 1e-122 | 14.7 | MA0079.1 (ACCCCGCCCC, -) | 0.00085 |
| 2-GATTGGC | 7 | 2049 | 1.6e-28 | 11.4 | MA0079.1 (GGGGCGGGGT, +) | 1.5 |
| 3-GTGACTCA | 8 | 1254 | 3e-21 | 12.5 | MA0079.1 (ACCCCGCCCC, -) | 1.5 |
| 4-GCGCNTGCGCA | 11 | 712 | 8.9e-16 | 17.1 | MA0079.1 (ACCCCGCCCC, -) | 1 |
| 5-GTGACGTCA | 9 | 930 | 4.2e-13 | 11.8 | MA0079.1 (GGGGCGGGGT, +) | 1 |
| 6-CCCGCCC | 7 | 3622 | 7.2e-11 | 14.0 | MA0079.1 (ACCCCGCCCC, -) | 0.0024 |
| 7-CTCCTCCTCC | 10 | 1894 | 0.00014 | 12.8 | MA0079.1 (ACCCCGCCCC, -) | 0.26 |
| 8-AAAAAAAAAAAAAAA | 15 | 564 | 0.031 | 12.5 | MA0039.1 (TAAAGGAAGG, +) | 0.0097 |
| 9-CACTTCCGG | 9 | 1263 | 5e-12 | 12.4 | MA0039.1 (CCTTCCTTTA, -) | 0.76 |
| 11-GAACTACAANTCCCA | 15 | 214 | 8.3e-07 | 21.5 | MA0079.1 (ACCCCGCCCC, -) | 0.25 |

## STREME motifs vs JASPAR SP1/KLF4

| motif | w | nsites | E | IC (bits) | best Tomtom match | E-value |
|---|---|---|---|---|---|---|
| 1-RGGGGCGGGGCYDG | 14 | 5374 | 8.6e-101 | 15.2 | MA0079.1 (GGGGCGGGGT, +) | 0.0011 |
| 2-YRRCCAATCRGMRV | 14 | 1668 | 3.2e-33 | 13.7 | MA0039.1 (TAAAGGAAGG, +) | 0.9 |
| 3-RTGASTCAY | 9 | 1183 | 5.9e-18 | 12.0 | MA0079.1 (GGGGCGGGGT, +) | 1.8 |
| 4-CACTTCCGGKT | 11 | 834 | 1.1e-11 | 12.8 | MA0039.1 (CCTTCCTTTA, -) | 0.22 |
| 5-CCCGCCCMC | 9 | 1896 | 2e-10 | 12.8 | MA0079.1 (ACCCCGCCCC, -) | 0.0019 |
| 6-WSTGCGCABGCGCRS | 15 | 675 | 2.1e-10 | 17.5 | MA0079.1 (GGGGCGGGGT, +) | 0.91 |
| 7-GTSACGTSAC | 10 | 586 | 1.5e-07 | 11.8 | MA0079.1 (GGGGCGGGGT, +) | 1.5 |
| 8-RACTACAAYTCCCAG | 15 | 227 | 1.7e-06 | 21.3 | MA0079.1 (ACCCCGCCCC, -) | 0.25 |
| 9-TGATTGACA | 9 | 247 | 0.0094 | 14.1 | MA0039.1 (TAAAGGAAGG, +) | 1.3 |
| 10-ATTTGCATAN | 10 | 148 | 0.021 | 14.4 | MA0039.1 (CCTTCCTTTA, -) | 0.69 |

## pystreme motifs vs STREME motifs

| motif | w | nsites | E | IC (bits) | best Tomtom match | E-value |
|---|---|---|---|---|---|---|
| 1-GGCCCCGCCCC | 11 | 5559 | 1e-122 | 14.7 | 1-RGGGGCGGGGCYDG (CAGGCCCCGCCCCC, -) | 4.1e-11 |
| 2-GATTGGC | 7 | 2049 | 1.6e-28 | 11.4 | 2-YRRCCAATCRGMRV (GCTCTGATTGGCTG, -) | 1.5e-06 |
| 3-GTGACTCA | 8 | 1254 | 3e-21 | 12.5 | 3-RTGASTCAY (GTGAGTCAC, +) | 8.6e-05 |
| 4-GCGCNTGCGCA | 11 | 712 | 8.9e-16 | 17.1 | 6-WSTGCGCABGCGCRS (CTGCGCCTGCGCAGT, -) | 4e-10 |
| 5-GTGACGTCA | 9 | 930 | 4.2e-13 | 11.8 | 7-GTSACGTSAC (GTGACGTCAC, +) | 2.8e-07 |
| 6-CCCGCCC | 7 | 3622 | 7.2e-11 | 14.0 | 5-CCCGCCCMC (CCCGCCCAC, +) | 0.00084 |
| 7-CTCCTCCTCC | 10 | 1894 | 0.00014 | 12.8 | 1-RGGGGCGGGGCYDG (CAGGCCCCGCCCCC, -) | 0.16 |
| 8-AAAAAAAAAAAAAAA | 15 | 564 | 0.031 | 12.5 | 8-RACTACAAYTCCCAG (AACTACAACTCCCAG, +) | 2.6 |
| 9-CACTTCCGG | 9 | 1263 | 5e-12 | 12.4 | 4-CACTTCCGGKT (CACTTCCGGTT, +) | 1.3e-07 |
| 11-GAACTACAANTCCCA | 15 | 214 | 8.3e-07 | 21.5 | 8-RACTACAAYTCCCAG (AACTACAACTCCCAG, +) | 7.5e-10 |
