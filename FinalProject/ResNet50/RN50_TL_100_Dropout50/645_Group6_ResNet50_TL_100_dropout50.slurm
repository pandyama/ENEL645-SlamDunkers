#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=/work/TALC/enel645_2022w/Software/miniconda3/bin:$PATH
source activate enel645

python /home/gregory.slowski/645-FinalProject/TL_100_Dropout50/645_Group6_ResNet50_TL_100_dropout50.py
conda deactivate
