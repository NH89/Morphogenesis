#!/bin/bash

#SBATCH --job-name=Morphogenesis_test1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=500m
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#Application specific commands:
echo "running Morphogenesis_test1" 
make_demo2 demo out
