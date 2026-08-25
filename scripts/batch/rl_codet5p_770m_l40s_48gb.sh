#!/bin/sh
#BSUB -q gpul40s
#BSUB -J rl_codet5p_770m_l40s
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o scripts/batch/logs/rl_codet5p_770m_l40s_%J.out
#BSUB -e scripts/batch/logs/rl_codet5p_770m_l40s_%J.err

set -eu

cd /work3/s204164/delta-proof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

RUN_NAME="${RUN_NAME:-rl_codet5p_770m_l40s_48gb_mixed_01}"

nvidia-smi
uv sync --frozen

set -- -m alphaproof.training.train "${RUN_NAME}" \
    --dtype mixed \
    --wandb-mode online

if [ -d "data/runs/${RUN_NAME}" ]; then
    echo "Resuming existing RL run ${RUN_NAME}."
    set -- "$@" --resume
else
    echo "Starting new RL run ${RUN_NAME}."
fi

set +e
uv run --no-sync python "$@"
training_status=$?
echo "Training command exited with status ${training_status} at $(date --iso-8601=seconds)."
nvidia-smi
