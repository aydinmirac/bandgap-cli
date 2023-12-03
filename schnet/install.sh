#!/bin/bash

# Create virtual env for SchNetPack
python3 -m venv ./schnet/venv

# activate virtual env
source ./schnet/venv/bin/activate

# install SchNetPack
pip install schnetpack
