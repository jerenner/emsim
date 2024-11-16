#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --partition=RTX_A4500
#SBATCH --mem=0
#SBATCH --nodelist=n0003.catscan0

source activate emsim

srun python main.py
# transformer.query_embeddings=1 \
# dataset.events_per_image_range=[1,1]
