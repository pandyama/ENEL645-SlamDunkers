#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=/work/TALC/enel645_2022w/Software/miniconda3/bin:$PATH
source activate enel645

python /home/makissin/EfficientNetB0-10-epochs-w-ft_dropout.py
conda deactivate
