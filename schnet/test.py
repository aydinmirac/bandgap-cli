################################################
######## Prediction of Some Structures #########
################################################

# Load libraries
import os
import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase as OMDB
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
import torch
import numpy as np
from ase import Atoms

# trained models store in the following directory. Assign it to a variable
omdb_folder = './omdb_folder'

## Some necessary parameters for test
batchsize = 48
cutoff = 5.
num_of_workers = 16


# Load the dataset
omdbdata = OMDB(
    datapath='./omdb_augmented_2023.db',
    batch_size=batchsize,
    num_train=42000,
    num_val=5415,
    num_test=5415,
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets(OMDB.BandGap, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units={OMDB.BandGap: 'eV'},
    num_workers=num_of_workers,
    split_file=os.path.join(omdb_folder, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=[OMDB.BandGap], #only load BandGap property
    #raw_path= './OMDB-GAP1_v1.1.tar.gz'
)

omdbdata.prepare_data()

# Load the best model
best_model = torch.load(os.path.join(omdb_folder, 'best_inference_model'), map_location=torch.device('cpu'))

# Show the results for trained model
#for batch in omdbdata.test_dataloader():
#    result = best_model(batch)
#    print("Result dictionary:", result)
#    break

# Create a converter for test structure
converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32)

# Create a test structure
"""
numbers = np.array([6, 1, 1, 1, 1])
positions = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                      [0.002150416, -0.0060313176, 0.0019761204],
                      [1.0117308433, 1.4637511618, 0.0002765748],
                      [-0.540815069, 1.4475266138, -0.8766437152],
                      [-0.5238136345, 1.4379326443, 0.9063972942]])
atoms = Atoms(numbers=numbers, positions=positions)
"""

# Import self-written python code to create atomic numbers and coordinates array from XYZ file
import aseObjects

file_name = "conf4.xyz"

numbers, positions = aseObjects.arrayCreation(file_name)

atoms = Atoms(numbers=numbers, positions=positions)

inputs = converter(atoms)

print('Keys:', list(inputs.keys()))

pred = best_model(inputs)

print('Prediction:', pred[OMDB.BandGap])

calculator = spk.interfaces.SpkCalculator(
    model_file=os.path.join(omdb_folder, "best_inference_model"), # path to model
    neighbor_list=trn.ASENeighborList(cutoff=cutoff), # neighbor list
    energy_key=OMDB.BandGap, # name of energy property in model
    energy_unit="eV", # units of energy property
    device="cpu", # device for computation
)
atoms.set_calculator(calculator)
print('Prediction:', atoms.get_total_energy(), "eV")