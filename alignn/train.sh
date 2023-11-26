#!/bin/zsh

# Source the ALIGNN venv
source ./alignn/venv/bin/activate

# Export the PATH
export PATH=$PATH:./alignn/alignn/alignn:./alignn/alignn/alignn/scripts

# Start training
train_folder.py --root_dir ./datasets/alignn --file_format cif --config ./alignn/config-1.json --output_dir=./alignn/output

# Deactivate the environment
deactivate
