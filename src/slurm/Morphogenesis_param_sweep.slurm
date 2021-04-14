#!/bin/bash
#
#SBATCH --job-name=Morphogenesis_map_params
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1
#
#SBATCH --array=0-53
#SBATCH --output=Morphogenesis_param_sweep_output.txt

FOLDER=($HOME/Programming/Cuda/Morphogenesis/data/demo_batch/demo_*/)

echo ${FOLDER[$SLURM_ARRAY_TASK_ID]}
srun make_demo2 ${FOLDER[$SLURM_ARRAY_TASK_ID]} ${FOLDER[$SLURM_ARRAY_TASK_ID]}

cat $FOLDER/results.csv > results_sweep.csv
tar -cf sweep_end_vtp.tar $FOLDER/*4099.vtp 
tar -cf sweep_end_csv.tar $FOLDER/*4099.csv 
tar -cf sweep_genome.tar $FOLDER/genome.csv
tar -cf sweep_Specfile.tar $FOLDER/SpecificationFile.txt
