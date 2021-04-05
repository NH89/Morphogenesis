#!/bin/bash

#SBATCH --job-name=Morphogenesis_test1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=500m

#Application specific commands:
echo "running Morphogenesis_test1" 
make_demo2 $HOME/Programming/Cuda/Morphogenesis/data/demo
