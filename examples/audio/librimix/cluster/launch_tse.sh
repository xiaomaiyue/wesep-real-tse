#!/bin/bash
# Autonomous launcher: waits until setup finishes (torch + editable wesep
# installed), tops up core training deps (covers a pesq/s3prl abort of the
# requirements install), then submits the 2-GPU training job. Runs detached
# in tmux so it survives SSH disconnect.
set -uo pipefail
PY="${HOME}/.conda/envs/wesep/bin/python"
PIP="${HOME}/.conda/envs/wesep/bin/pip"
M="-i https://pypi.tuna.tsinghua.edu.cn/simple"
LOG="${HOME}/launch_tse.log"

say(){ echo "[launch $(date +%H:%M:%S)] $*"; }

say "waiting for setup to finish (torch importable + wesep installed)..."
n=0
until "$PY" -c "import torch, importlib.util as u, sys; sys.exit(0 if u.find_spec('wesep') else 1)" 2>/dev/null; do
  n=$((n+1))
  if [ $n -gt 180 ]; then say "GAVE UP after ~60min: setup never finished. Check tmux 'setup' window."; exit 1; fi
  sleep 20
done
say "env ready: torch $("$PY" -c 'import torch;print(torch.__version__)')"

say "topping up core training deps (idempotent)..."
"$PIP" install $M "numpy<2" scipy soundfile librosa pyyaml tqdm h5py kaldiio lmdb fire \
  fast_bss_eval auraloss torchmetrics matplotlib tableprint joblib mir_eval pystoi >/dev/null 2>&1 || say "WARN: some top-up deps failed"
( cd "${HOME}/tse1/wesep-real-tse" && "$PIP" install -e . --no-deps >/dev/null 2>&1 || say "WARN: editable install hiccup" )

# final import sanity (training-critical modules)
if ! "$PY" -c "import torch, soundfile, yaml, fast_bss_eval, wesep" 2>/dev/null; then
  say "ERROR: core imports still failing; NOT submitting. Inspect: $LOG"
  "$PY" -c "import torch, soundfile, yaml, fast_bss_eval, wesep" 2>&1 | tail -5
  exit 1
fi

say "submitting training job..."
cd "${HOME}/tse1/wesep-real-tse/examples/audio/librimix/cluster"
sbatch train_text_tse.slurm | tee "${HOME}/tse_job.txt"
JID=$(awk '{print $NF}' "${HOME}/tse_job.txt")
say "submitted job ${JID}. log -> $(pwd)/tse_text-${JID}.out"
sleep 5
squeue -j "${JID}" 2>/dev/null || true
say "DONE."
