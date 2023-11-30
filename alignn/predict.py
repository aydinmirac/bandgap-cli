import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch
import sys
from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Pretrained Models"
)
parser.add_argument(
    "--checkpoint_file",
)

parser.add_argument(
    "--file_format", default="cif", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--file_path",
    default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    help="Path to file.",
)

parser.add_argument(
    "--cutoff",
    default=5,
    help="Distance cut-off for graph constuction"
    + ", usually 8 for solids and 5 for molecules.",
)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_prediction(
    checkpoint_file,
    atoms=None,
    cutoff=5,
):

    model = ALIGNN(ALIGNNConfig(name="alignn", output_features=1))
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)["model"])
    model.to(device)
    model.eval()
    
    #if os.path.exists(filename):
    #    os.remove(filename)

    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff))
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    checkpoint_file = args.checkpoint_file
    file_path = args.file_path
    file_format = args.file_format
    cutoff = args.cutoff
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", file_format)

    out_data = get_prediction(
        checkpoint_file=checkpoint_file, cutoff=float(cutoff), atoms=atoms)

    print("Predicted value:", file_path, out_data)
