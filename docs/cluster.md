# Running on a cluster

pystreme was developed on an LSF cluster with GPU nodes. None of this is
needed on a workstation with a GPU; on a shared cluster, these are the
things that went wrong in practice.

## Rules of thumb

- **Run through the scheduler, never on a login node.** `fit` belongs on a
  GPU node; tests and small scripts on a CPU node.
- **Request memory explicitly.** Importing torch alone takes a few GB, more
  than many queues grant by default.
- **`/tmp` is usually node-local.** Scripts and data a job needs must live on
  the shared filesystem.
- **Keep your own `~/.local` packages out.** Packages installed with
  `pip install --user` shadow the environment's own; set
  `PYTHONNOUSERSITE=1` when building the environment and in every job
  (`scripts/setup_env.sh` does both).

## LSF

```bash
# CPU: the test suite
bsub -q <cpu-queue> -K -R "rusage[mem=6000]" -M 6000 -oo tests.log \
  "export PYTHONNOUSERSITE=1; cd pystreme; python -m pytest -q"

# GPU: an analysis
bsub -q <gpu-queue> -gpu "num=1" -K -R "rusage[mem=16000]" -M 16000 -oo run.log \
  "export PYTHONNOUSERSITE=1; python find_motifs.py peaks.bed genome.fa results"
```

`-K` blocks until the job ends and returns its exit code. The `-oo` log is
written when the job finishes.

## SLURM

```bash
sbatch --gres=gpu:1 --mem=16G --output=run.log \
  --wrap "export PYTHONNOUSERSITE=1; python find_motifs.py peaks.bed genome.fa results"
```

## The MEME suite and the validation scripts

`scripts/validate_streme.py` and `scripts/validate_fimo.py` call `streme`,
`tomtom` and `fimo`. If the MEME suite comes from an environment module,
loading it can put the module's own Python first on `PATH`, ahead of your
environment's. Call your environment's interpreter by its absolute path:

```bash
module load MEME
PY=/path/to/envs/pystreme/bin/python
export PYTHONNOUSERSITE=1
$PY scripts/validate_streme.py --peaks peaks.bed --genome genome.fa --device cuda --out validation/run
```

## Timing

GPU models differ by roughly 1.5x in speed, and a card shared with other
jobs can be off by 2x in either direction. When timing matters, reserve the
GPU for the job (on LSF: `-gpu "num=1:j_exclusive=yes"`).
