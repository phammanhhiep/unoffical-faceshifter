#!/bin/bash

DEFAULT_CONDA_ENV="ct2021_2"

myenv="$DEFAULT_CONDA_ENV"

eval "$(conda shell.bash hook)"
conda activate "$myenv"

python "aei_trainer.py" -c config/train.yaml --checkpoint "$1"