#!/bin/sh
#BSUB -q gpuv100
#BSUB -J batch_sizes_770m_v100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o scripts/batch/logs/batch_sizes_770m_%J.out
#BSUB -e scripts/batch/logs/batch_sizes_770m_%J.err

set -eu

cd /work3/s204164/delta-proof
mkdir -p data/benchmarks scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

nvidia-smi
uv sync --frozen
uv run --no-sync python scripts/benchmark_rl_batch_sizes.py \
    --output data/benchmarks/rl_batch_sizes_770m_v100_32gb.json
nvidia-smi
