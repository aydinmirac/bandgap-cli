#!/bin/zsh

# Create virtual env for ALIGNN
python3 -m venv ./schnet/venv

# activate virtual env
source ./schnet/venv/bin/activate

# install ALIGNN
pip install schnetpack