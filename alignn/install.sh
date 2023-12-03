#!/bin/bash

# Download ALIGNN git repository
git clone https://github.com/usnistgov/alignn.git ./alignn/main/

# Create virtual env for ALIGNN
python3 -m venv ./alignn/venv

# activate virtual env
source ./alignn/venv/bin/activate

# install ALIGNN
pip install alignn
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install urllib3==1.26.6
