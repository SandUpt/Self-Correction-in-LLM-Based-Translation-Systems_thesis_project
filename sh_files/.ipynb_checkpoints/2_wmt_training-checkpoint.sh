#!/bin/bash

LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE_OUT="$LOG_DIR/2_wmt_training_${TS}.log"
LOG_FILE_ERR="$LOG_DIR/2_wmt_training_${TS}.err"


exec > >(tee -a "$LOG_FILE_OUT")    
exec 2> >(tee -a "$LOG_FILE_ERR" >&2)  

echo "Using GPU inside JupyterLab session"
echo "stdout log: $LOG_FILE_OUT"
echo "stderr log: $LOG_FILE_ERR"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"

nvidia-smi

cd /sc/home/sandeep.uprety/thesis_project/self_correction_llm_based_translation_thesis/thesis_project/scripts/ || exit 1

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 2_wmt_training.py

echo "Done at: $(date)"
