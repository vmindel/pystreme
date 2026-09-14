# pystreme

[![tests](https://github.com/vmindel/pystreme/actions/workflows/tests.yml/badge.svg)](https://github.com/vmindel/pystreme/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-vmindel.github.io%2Fpystreme-0e6b60)](https://vmindel.github.io/pystreme/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22749207.svg)](https://doi.org/10.5281/zenodo.22749207)

De novo motif discovery from peak sets: the STREME algorithm, batched on a
GPU and called from Python. Give it a BED file and a genome; get back each
motif with its sites, its position in the peaks, and its closest known
transcription factor. About a minute for ten motifs in 7,000 peaks.

> **Please read before using**
>
> - **Unofficial.** An independent reimplementation of STREME
>   ([Bailey, *Bioinformatics* 2021](https://doi.org/10.1093/bioinformatics/btab203)),
>   written from the paper. It is not affiliated with or endorsed by the
>   MEME Suite. If you use pystreme, please cite STREME.
> - **Written by an AI.** The code was written by Claude (Anthropic) using
>   Claude Code. The design, the validation against STREME and FIMO on real
>   ChEC-seq data, and every decision about what counts as correct were
>   directed and checked by Vladimir Mindel
>   ([@vmindel](https://github.com/vmindel)). The full
>   validation record is in [`BENCHMARKS.md`](BENCHMARKS.md).
> - **A first-look tool.** It is built for fast quality control of a peak
>   set. Confirm anything you intend to report with established tools (the
>   MEME Suite, HOMER, or a deep-learning model).
> - No warranty; see [`LICENSE`](LICENSE).

## What it does

- **STREME's round loop:** a dinucleotide-shuffled control (or GC-matched,
  or your own), a held-out test set, k-mer seeds refined into motifs, each
  motif's threshold re-tested on peaks it never saw, then its sites erased
  before the next round.
- **Positions for free:** every site's offset from the peak centre, and a
  central-enrichment test that tends to separate the factor you pulled down
  from its co-factors.
- **Everything in Python:** `Motif` objects, pandas tables, database
  annotation (e.g. HOCOMOCO), sequence logos, BED and MEME-format export.
  No subprocess, no HTML to parse.
- **Useful pieces on their own:** a batched PWM scanner and a SEA-style
  enrichment test for known motifs.

## Install

pystreme needs Python ≥ 3.10 and PyTorch. Install the PyTorch build that
matches your GPU first ([pytorch.org](https://pytorch.org/get-started/locally/)),
then:

```bash
pip install "pystreme[io,tables,plot] @ git+https://github.com/vmindel/pystreme"
```

To reproduce the exact environment it was validated in (Python 3.10, torch
2.2.2 + CUDA 12.1), use [`environment.yml`](environment.yml) and
[`requirements-lock.txt`](requirements-lock.txt), or run
`scripts/setup_env.sh`.

## Quick start

```python
from pystreme import MotifDiscovery, summary, annotate, plot

disc = MotifDiscovery.from_bed("peaks.bed", "genome.fa", size=200, device="cuda", seed=0)
motifs = disc.fit(n_motifs=10, verbose=True)

summary(motifs)                                      # one row per motif
ann = annotate(motifs, "H12CORE_meme_format.meme")   # closest known factors
plot.report(motifs, annotation=ann)                  # logos + position histograms
disc.to_bed(motifs, "sites.bed")                     # every site, genome coordinates
disc.to_meme(motifs, "motifs.meme")                  # for Tomtom, FIMO, HOMER
```

Peaks should be summit-centred; `size=200` takes 200 bp around each peak
centre. The genome must be an indexed FASTA (`.fai` alongside). Motif
databases are not bundled: download one in MEME format, for example
HOCOMOCO v12 or JASPAR.

**Quote `holdout_logp`**, the p-value re-tested on held-out peaks. The
training `log_pvalue` is optimistic by construction. `central_logp`
measures how strongly sites pile up at the peak centre.

The documentation site, **<https://vmindel.github.io/pystreme/>**, has the
full [usage guide](docs/usage.md) (every argument, the controls, how to read
the results) and an API reference generated from the docstrings. On a shared cluster, see
[running on a cluster](docs/cluster.md).

## Does it work?

Measured against the real tools, on 7,232 SP1 ChEC-seq peaks from mouse
(200 bp, motif widths 6–15, ten motifs):

| | pystreme | reference |
|---|---|---|
| motifs also found by STREME 5.5.0 | 8 of 10, including the primary SP1 motif | |
| wall-clock | **62 s** on one A40 GPU | STREME: 191 s |
| scanner scores vs FIMO | Pearson r = 1.00000, 99.95% identical best sites | |

Run `fit` on a GPU: on a CPU it is several times slower. How these numbers
were measured, and what changed along the way, is in
[`BENCHMARKS.md`](BENCHMARKS.md). The design and its trade-offs are in
[`DESIGNDOC.md`](DESIGNDOC.md).

## Tests

```bash
pytest            # CPU only, no GPU needed
pytest -m gpu     # the subset that needs a CUDA device
```

`tests/test_real_data.py` also runs on real data when you point
`PYSTREME_TEST_GENOME` and `PYSTREME_TEST_PEAKS` at a genome and an SP/KLF
peak set (ChIP- or ChEC-seq).

## Repository layout

- `src/pystreme/`: the package.
- `docs/` + `mkdocs.yml`: the documentation site (`pip install -r docs/requirements.txt`, then `mkdocs serve`).
- `scripts/validate_streme.py`, `scripts/validate_fimo.py`: the head-to-head comparisons against STREME and FIMO (need the MEME Suite).
- `notebooks/smoke_test.ipynb`: the whole flow in a notebook.
- `skills/pystreme/`: a [Claude Code](https://claude.com/claude-code) skill that teaches an agent the API and its pitfalls (`ln -s $PWD/skills/pystreme ~/.claude/skills/pystreme`).

## Citing

If pystreme helps your work, please cite STREME, whose algorithm it
implements:

> Bailey TL. STREME: accurate and versatile sequence motif discovery.
> *Bioinformatics* 37(18):2834–2840 (2021).
> [doi:10.1093/bioinformatics/btab203](https://doi.org/10.1093/bioinformatics/btab203)

and pystreme itself: [doi:10.5281/zenodo.22749207](https://doi.org/10.5281/zenodo.22749207) (all versions;
each release also has its own DOI on Zenodo), or see
[`CITATION.cff`](CITATION.cff).

## License

MIT. The two JASPAR motifs used as test fixtures are CC BY 4.0; see
[`tests/fixtures/README.md`](tests/fixtures/README.md).
