#!/usr/bin/env bash

export CONDA_ENV_NAME=3d_pose-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

pip install numpy==1.21.5 torch==1.4.0 torchvision==0.5.0
pip install -r requirements.txt