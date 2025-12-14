#!/bin/bash -eux
#SBATCH --job-name=8_self_correction_evaluation_2
#SBATCH --account=sci-demelo-student
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gpus=a100:1
#SBATCH --time=20:00:00
#SBATCH --output=../logs/8_self_correction_evaluation_2_%j.log
#SBATCH --error=../logs/8_self_correction_evaluation_2_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

nvidia-smi

cd ~/thesis_project/self_correction_llm_based_translation_thesis/thesis_project/scripts/

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "8_self_correction_evaluation_2"
python3 8_self_correction_evaluation_2.py

echo "Done at: $(date)"
