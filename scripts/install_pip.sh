#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv 3d_pose-env
echo "Activating virtual environment"

source $PWD/3d_pose-env/bin/activate

$PWD/3d_pose-env/bin/pip install numpy==1.21.5 torch==1.4.0 torchvision==0.5.0
$PWD/3d_pose-env/bin/pip install -r requirements.txt