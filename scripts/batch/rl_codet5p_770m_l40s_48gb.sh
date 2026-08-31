#!/bin/sh
#BSUB -q gpul40s
#BSUB -J rl_codet5p_770m_l40s
#BSUB -n 48
#BSUB -R "select[hname!='n-62-13-18']"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o data/runs/rl_codet5p_770m_l40s_48gb_04/lsf_%J.out
#BSUB -e data/runs/rl_codet5p_770m_l40s_48gb_04/lsf_%J.err

set -eu

cd /work3/s204164/delta-proof
module purge
module load python3/3.13.11
module load cuda/12.6.3

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

RUN_NAME="rl_codet5p_770m_l40s_48gb_04"
mkdir -p "data/runs/${RUN_NAME}"

nvidia-smi
uv sync --frozen

if [ -f "data/runs/${RUN_NAME}/config.json" ]; then
    echo "Resuming existing RL run ${RUN_NAME}."
    set -- -m alphaproof.training.rl_cli "${RUN_NAME}" --resume
else
    echo "Starting new RL run ${RUN_NAME}."
    set -- -m alphaproof.training.rl_cli \
        "${RUN_NAME}" \
        alphaproof/yaml/codet5p_770m_l40s.yaml
fi

set +e
uv run --no-sync python "$@"
training_status=$?
echo "Training command exited with status ${training_status} at $(date --iso-8601=seconds)."
nvidia-smi
exit "${training_status}"
