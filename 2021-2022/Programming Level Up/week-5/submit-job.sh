#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --job-name=mnist
#SBATCH --output=train.mnist.out
#SBATCH --error=train.mnist.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:1

~/miniconda3/envs/testing/bin/python test-script.py
