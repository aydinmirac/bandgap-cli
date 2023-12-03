#!/bin/bash

# Source the ALIGNN venv
source ./alignn/venv/bin/activate

# Export the PATH
export PATH=$PATH:./alignn/alignn/alignn:./alignn/alignn/alignn/scripts

# Create datetime variable for output file
current_date=$(date +"%Y-%m-%d_%H:%M:%S")

# Get arguments from terminal. These arguments will come from "test_alignn" commands
best_model=$1
file_format=$2
file_path=$3
cutoff=$4

# Start testing
python ./alignn/predict.py --checkpoint_file $best_model --file_format $file_format --file_path $file_path --cutoff $cutoff > ./predictions/alignn_result_$current_date.txt

# Deactivate the environment
deactivate
