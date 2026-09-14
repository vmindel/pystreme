#!/usr/bin/env bash
# Create (or recreate) the pystreme conda env, fully isolated from any
# ~/.local user-site packages, and do an editable install of the package.
#
# Why PYTHONNOUSERSITE matters here: on this cluster, Python's user-site
# directory (~/.local/lib/pythonX.Y/site-packages) takes precedence over a
# conda/mamba env's own site-packages on sys.path. If you have anything
# installed there via `pip install --user` (common on shared clusters), pip
# will report packages "already satisfied" from user-site during env
# creation and silently skip installing them into the env itself -- so the
# env *imports* fine for you, but is missing real dependencies (this bit us
# with torch's own CUDA runtime libs: nvidia-cublas-cu12 etc. were "already
# satisfied" from user-site and never installed into the env). Exporting
# PYTHONNOUSERSITE=1 before `micromamba create` makes its internal pip calls
# ignore user-site, so what actually gets installed matches what a clean
# account (no ~/.local cruft) would get -- which is the whole point of a
# locked env meant to be reproducible for other people.
#
# Usage:
#   ./scripts/setup_env.sh
#
# Requires micromamba (or conda) on PATH, and a repo checkout as cwd or
# passed as $1.

set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${PYSTREME_ENV_PREFIX:-$(dirname "$(dirname "$(command -v conda 2>/dev/null || echo /opt/conda/bin/conda)")")/envs/pystreme}"
# Fallback: if conda isn't found, still default to a sane location next to
# wherever micromamba's base conda install would be; override via
# PYSTREME_ENV_PREFIX if this guess is wrong for your setup.

echo "Repo:        $REPO_ROOT"
echo "Env prefix:  $ENV_PREFIX"

if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba not found on PATH -- see environment.yml and adapt to 'conda env create' if needed." >&2
    exit 1
fi

echo "--- creating env (user-site isolated) ---"
PYTHONNOUSERSITE=1 micromamba create -p "$ENV_PREFIX" -f "$REPO_ROOT/environment.yml" -y

echo "--- adding permanent user-site isolation hook (survives future activations) ---"
mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
#!/bin/sh
export _PYSTREME_OLD_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-__unset__}"
export PYTHONNOUSERSITE=1
EOF
cat > "$ENV_PREFIX/etc/conda/deactivate.d/env_vars.sh" <<'EOF'
#!/bin/sh
if [ "${_PYSTREME_OLD_PYTHONNOUSERSITE:-}" = "__unset__" ]; then
    unset PYTHONNOUSERSITE
else
    export PYTHONNOUSERSITE="${_PYSTREME_OLD_PYTHONNOUSERSITE}"
fi
unset _PYSTREME_OLD_PYTHONNOUSERSITE
EOF
chmod +x "$ENV_PREFIX/etc/conda/activate.d/env_vars.sh" "$ENV_PREFIX/etc/conda/deactivate.d/env_vars.sh"

echo "--- editable install of pystreme ---"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/pip" install -e "$REPO_ROOT[io,plot,dev]"

echo "--- sanity check ---"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" -c "
import sys
assert not any('.local' in p for p in sys.path), 'user-site leaked into the env!'
import torch, pystreme
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('pystreme', pystreme.__version__)
"

echo "--- regenerating requirements-lock.txt ---"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/pip" freeze --exclude-editable > "$REPO_ROOT/requirements-lock.txt"

echo "Done. Activate with: conda activate pystreme   (or micromamba activate pystreme)"
