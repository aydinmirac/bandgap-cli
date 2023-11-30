#!/bin/zsh

# Download ALIGNN git repository
git clone https://github.com/usnistgov/alignn.git ./alignn/main/

# Create virtual env for ALIGNN
python3 -m venv ./alignn/venv

# activate virtual env
source ./alignn/venv/bin/activate

# install ALIGNN
pip install alignn