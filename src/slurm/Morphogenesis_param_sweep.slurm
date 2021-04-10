#!/bin/bash
#
#SBATCH --job-name=Morphogenesis_map_params
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1
#
#SBATCH --array=0-2
#SBATCH --output=${FOLDER[$SLURM_ARRAY_TASK_ID]}Morphogenesis_map_params_output.txt

FOLDER=($HOME/Programming/Cuda/Morphogenesis/data/demo_batch/demo_intstiff*.000000/)

echo ${FOLDER[$SLURM_ARRAY_TASK_ID]}
srun make_demo2 ${FOLDER[$SLURM_ARRAY_TASK_ID]} ${FOLDER[$SLURM_ARRAY_TASK_ID]}
