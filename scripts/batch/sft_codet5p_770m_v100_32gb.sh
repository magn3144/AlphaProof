#!/bin/sh
#BSUB -q gpuv100
#BSUB -J sft_codet5p_770m_v100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o scripts/batch/logs/sft_codet5p_770m_%J.out
#BSUB -e scripts/batch/logs/sft_codet5p_770m_%J.err

set -eu

cd /work3/s204164/mini-alphaproof
mkdir -p scripts/batch/logs

module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-sft_codet5p_770m_v100_32gb}"

nvidia-smi

uv sync --frozen

if [ -d "data/runs/${RUN_NAME}" ]; then
    echo "Resuming existing SFT run ${RUN_NAME}."
    set -- -m alphaproof.training.sft "${RUN_NAME}" --resume
else
    echo "Starting new SFT run ${RUN_NAME}."
    set -- -m alphaproof.training.sft \
        "${RUN_NAME}" \
        alphaproof/yaml/codet5p_770m_l40s.yaml
fi

uv run --no-sync python "$@"
