import subprocess
import typer
import os
from utils import copy_files, edit_csv_files

#Create the app
app = typer.Typer()

@app.command()
def copy_data():
    """
    The command copies your molecules to SchNetPack and ALIGNN data directories.
    """
    source_dir = './data_copy/test'
    destination_dir = './data_copy/destination'
    source_csv = './data_copy/test/id_prop.csv'
    destination_csv = './data_copy/destination/id_prop.csv'
    copy_files(source_dir, destination_dir)
    edit_csv_files(source_csv, destination_csv) 

@app.command()
def train_schnet():
    pass

@app.command()
def test_schnet():
    pass

@app.command()
def train_alignn():
    subprocess.run(["./alignn/train.sh"])

@app.command()
def test_alignn():
    pass

if __name__ == "__main__":
    app()