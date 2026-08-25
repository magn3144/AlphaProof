#!/bin/sh
#BSUB -q gpuv100
#BSUB -J rl_codet5p_220m_v100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o scripts/batch/logs/rl_codet5p_220m_%J.out
#BSUB -e scripts/batch/logs/rl_codet5p_220m_%J.err

set -eu

cd /work3/s204164/delta-proof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

RUN_NAME="${RUN_NAME:-rl_codet5p_220m_v100_32gb_05}"

nvidia-smi
uv sync --frozen

set -- -m alphaproof.training.train \
    "${RUN_NAME}" \
    --dataset-dir data/dataset/numina_math_lean_passing \
    --num-simulations 64 \
    --num-games 250 \
    --rollout-max-action-length 32 \
    --batch-size 20 \
    --learning-rate 1e-5 \
    --training-steps 200 \
    --training-iterations 200 \
    --checkpoint-interval 50 \
    --value-weight 0.01 \
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
exit "${training_status}"
