#!/bin/bash
#SBATCH -J tfb
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
module load anaconda3/2019.07 cuda/9.0 cudnn/7.3.0
source activate tf1.12

python tf_bert_dense.py