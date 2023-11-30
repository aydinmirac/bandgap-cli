#!/bin/zsh

# Source the ALIGNN venv
source ./alignn/venv/bin/activate

# Export the PATH
export PATH=$PATH:./alignn/main/alignn:./alignn/main/alignn/scripts

# Create datetime variable for output file
current_date=$(date +"%Y-%m-%d_%H:%M:%S")

# Start training
train_folder.py --root_dir ./datasets/alignn --file_format cif --config ./alignn/config.json --output_dir=./alignn/output_$current_date

# Deactivate the environment
deactivate
