#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval_sft_codet5p_770m
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o scripts/batch/logs/eval_sft_codet5p_770m_%J.out
#BSUB -e scripts/batch/logs/eval_sft_codet5p_770m_%J.err

set -eu

cd /work3/s204164/mini-alphaproof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi

uv run --no-sync python -m scripts.evaluate_sft_checkpoints \
    data/runs/sft_codet5p_770m_v100_32gb \
    data/dataset/numina_sft_evaluation/test.jsonl \
    data/evaluations/sft_codet5p_770m_numina_30 \
    --num-simulations 32
