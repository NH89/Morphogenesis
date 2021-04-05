#!/bin/sh

#SBATCH --job-name=Morphogenesis_map_params
#SBATCH --time=01:00:00
#SBATCH --mem=500m
i=1
j=3
k=2
m=3
#SBATCH --nodes=i
#SBATCH --tasks-per-node=j  (or --ntasks-per-node=j)
#SBATCH --cpus-per-task=k
#SBATCH --gres=gpu:m

for FOLDER in HOME/Programming/Cuda/Morphogenesis/data/demo_intstiff*; 
do
make_demo2 FOLDER
done

# sbatch --export IN=$FILE myjob.q; 
#for FILE in *.in; do sbatch --export IN=$FILE myjob.q; done
#file:///home/nick/Programming/Cuda/Morphogenesis/data/demo_intstiff1.000000

#$HOME/Programming/Cuda/Morphogenesis/data/demo
