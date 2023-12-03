import subprocess
import typer
from typing_extensions import Annotated
import os
import json
from utils import copy_files, edit_csv_files, edit_json_file


#Create the app
app = typer.Typer(pretty_exceptions_short=False)

@app.command()
def install_alignn():
    """
    This command installs ALIGNN package.
    """
    subprocess.run(['./alignn/install.sh'], check=True)    

@app.command()
def install_schnet():
    """
    This command installs SchNetPack package.
    """
    subprocess.run(['./schnet/install.sh'], check=True)  

@app.command()
def extract_dataset():
    """
    This command extracts necessary dataset for both ALIGNN and SchNetPack.
    """
    subprocess.run(["./datasets/extract.sh"], check=True)

@app.command()
def train_schnet(batchsize: Annotated[str, typer.Argument(help="The batch size")] = "32", 
            cutoff: Annotated[str, typer.Argument(help="The cutoff ratio")] = "5.0",
            num_of_workers: Annotated[str, typer.Argument(help="The number of workers for data load")] = "4",
            n_atom_basis: Annotated[str, typer.Argument(help="The number of features to describe atomic environments")] = "128",
            n_gaussians: Annotated[str, typer.Argument(help="The number of Gaussian fuctions")] = "50",
            n_interactions: Annotated[str, typer.Argument(help="The number of interaction blocks")] = "3",
            lr_rate: Annotated[str, typer.Argument(help="The learning rate")] = "0.01", 
            n_of_epochs: Annotated[str, typer.Argument(help="The epoch size")] = "5", 
            train_size: Annotated[str, typer.Argument(help="The size of training set")] = "1000", 
            val_size: Annotated[str, typer.Argument(help="The size of validation set")] = "100",
            test_size: Annotated[str, typer.Argument(help="The size of test set")] = "100"):
    """
    This command starts a training process by using SchNetPack model.
    """
    with open('errors.log', 'w') as err_file:
        output = subprocess.run(["./schnet/venv/bin/python","./schnet/train.py", 
            batchsize, cutoff, num_of_workers, n_atom_basis, n_gaussians, n_interactions, lr_rate, n_of_epochs, 
            train_size, val_size, test_size], check=True, stderr=err_file)
    

@app.command()
def test_schnet(model_folder: Annotated[str, typer.Argument(help="The folder name that includes 'best_inference_model' from your last run")] = "./schnet/results_1",
            molecule_file: Annotated[str, typer.Argument(help="The CIF file that you want to predict")] = "./predictions/1008775.cif",
            cutoff: Annotated[str, typer.Argument(help="The cutoff ratio")] = "5.0"):
    """
    This command predicts the bandgap of a single molecule with the trained SchNetPack model.
    """
    with open('errors.log', 'w') as err_file:
        output = subprocess.run(["./schnet/venv/bin/python","./schnet/test.py",
            model_folder, molecule_file, cutoff], check=True, stderr=err_file)
    

@app.command()
def copy_data():
    """
    This command copies your molecules to SchNetPack and ALIGNN data directories.
    """
    source_dir = './copy-data/'
    destination_dir = './datasets/main'
    source_csv = './copy-data/id_prop.csv'
    destination_csv = './datasets/main/id_prop.csv'
    copy_files(source_dir, destination_dir)
    edit_csv_files(source_csv, destination_csv) 

@app.command()
def create_database():
    """
    This command creates a ASE database for SchNetPack model.
    """
    subprocess.run(["./schnet/venv/bin/python","./schnet/database.py"], check=True)

@app.command()
def train_alignn(n_val: Annotated[int, typer.Argument(help="The size of validation set")] = 100, 
            n_test: Annotated[int, typer.Argument(help="The size of test set")] = 100, 
            n_train: Annotated[int, typer.Argument(help="The size of train set")] = 100, 
            epochs: Annotated[int, typer.Argument(help="The epoch size")] = 10, 
            batch_size: Annotated[int, typer.Argument(help="The batch size")] = 20, 
            learning_rate: Annotated[float, typer.Argument(help="learning rate in float")] = 0.01, 
            num_workers: Annotated[int, typer.Argument(help="The number of workers for data load")] = 4, 
            cutoff: Annotated[float, typer.Argument(help="The cutoff ratio. Please check Schetpack or Alignn document for it")] = 5.0, 
            max_neighbors: Annotated[int, typer.Argument(help="The maximum number of neighbours. Please check Schetpack or Alignn document for it")] = 8):

    """
    This command starts a training process by using ALIGNN model.
    """
    # Edit JSON file for ALIGNN training
    edit_json_file(n_val, n_test, n_train, epochs, batch_size, learning_rate, num_workers, cutoff, max_neighbors)
    # Start training process
    subprocess.run(["./alignn/train.sh"])

@app.command()
def test_alignn(best_model: Annotated[str, typer.Argument(help="The path of best model from your last training, stored in the last output folder as best_model.pt")],
                file_format: Annotated[str, typer.Argument(help="The file format, currently only cif supported")] = "cif",
                molecule_path: Annotated[str, typer.Argument(help="The path of your test molecule, please store it under 'predictions' folder")] = "./predictions/1008775.cif",
                cutoff: Annotated[str, typer.Argument(help="The size of validation set")] = "5.0"):
    """
    This command predicts the bandgap of a single molecule with the trained ALIGNN model.
    """
    # Start testing process
    subprocess.run(["./alignn/test.sh", best_model, file_format, molecule_path, cutoff], check=True)

if __name__ == "__main__":
    app()
