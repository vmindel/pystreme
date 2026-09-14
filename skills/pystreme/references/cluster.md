# Running pystreme on a shared cluster (LSF / SLURM)

## Rule zero

On a shared cluster, **no computation on the login node** — not `pytest`,
not a smoke script, not `python -c "import torch"`. The login node is for
editing, git, and reading logs; everything else goes through the scheduler.
Ask the user before launching jobs.

## Recipes

LSF, CPU (tests, small scripts):

```bash
bsub -q <cpu-queue> -K -R "rusage[mem=6000]" -M 6000 -oo run.log \
  "export PYTHONNOUSERSITE=1; cd <repo>; <env>/bin/python -m pytest -q"
```

LSF, GPU (`fit`, benchmarks, the notebook):

```bash
bsub -q <gpu-queue> -gpu "num=1" -K -R "rusage[mem=16000]" -M 16000 -oo run.log \
  "export PYTHONNOUSERSITE=1; cd <repo>; <env>/bin/python my_run.py"
```

SLURM: `sbatch --gres=gpu:1 --mem=16G --output=run.log --wrap "..."`.

From an agent: submit a blocking job (`bsub -K`, or `srun`) through Bash in
the background, then read the log file.

## Gotchas (all hit in practice)

- Memory must be explicit: a queue's default (often 1 GB) kills a bare
  torch import.
- `/tmp` is node-local: a script written there on the login node does not
  exist on the compute node. Keep scratch scripts on the shared filesystem.
- A MEME-suite environment module can put its own Python first on `PATH`.
  Call the env's interpreter by absolute path, with `PYTHONNOUSERSITE=1`.
- `~/.local` user-site packages shadow the env unless `PYTHONNOUSERSITE=1`.
- Job logs are usually written when the job ends; there is no partial
  output to poll.
- Timings: GPU models differ ~1.5x; a shared card can be off 2x either way.
  Reserve the card (`-gpu "num=1:j_exclusive=yes"` on LSF) for benchmarks.
