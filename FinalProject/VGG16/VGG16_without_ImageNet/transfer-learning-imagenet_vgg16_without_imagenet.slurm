#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=/work/TALC/enel645_2022w/Software/miniconda3/bin:$PATH
source activate enel645

python /home/zachary.frena/TL-garbage-classification/transfer_learning_imagenet_vgg16_without_imagenet.py
conda deactivate
