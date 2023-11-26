import os
import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase as OMDB
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

omdb_folder = './omdb_folder'
if not os.path.exists('omdb_folder'):
    os.makedirs(omdb_folder)

from ase.db import connect
import numpy as np
import time

"""
# If there is a problem loading the db, use the snippet below:

db_path_new = "./omdb_fixed.db"
with connect(db_path) as db:
    with connect(db_path_new) as db_new:
        db_new.metadata = db.metadata
        for idx in range(len(db)):
            atmsrw = db.get(idx+1)
            data = {property_name: np.array([property]) for property_name, property in atmsrw.data.items()}
            db_new.write(atmsrw.toatoms(), data=data)
"""

## Parameters
db_path = "./omdb_augmented_2023.db"
batchsize = 40
cutoff = 10.
num_of_workers = 0
n_atom_basis = 128
n_gaussians = 50
n_interactions = 3
lr_rate = 1e-3
n_of_epochs = 5
train_size = 1000
val_size = 1000
test_size = 1000

# Add timer
start_time = time.time()

# Increase Torch precision
torch.set_float32_matmul_precision("medium")

## Print parameters
print("#############################")
print("HYPERPARAMETERS IN THIS MODEL")
print("#############################")
print("Batchsize:", batchsize)
print("cutoff:", cutoff)
print("Number of workers:", num_of_workers)
print("Number of interaction layers:", n_interactions)
print("Learning rate:", lr_rate)
print("Number of Gaussians:", n_gaussians)
print("Number of atomwise features:", n_atom_basis)
print("Number of epochs:", n_of_epochs)

omdbdata = OMDB(
    datapath=db_path,
    batch_size=batchsize,
    num_train=train_size,
    num_val=val_size,
    num_test=test_size,
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets(OMDB.BandGap, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    property_units={OMDB.BandGap: 'eV'},
    num_workers=num_of_workers,
    split_file=os.path.join(omdb_folder, "split.npz"),
    pin_memory=False, # set to false, when not using a GPU
    load_properties=[OMDB.BandGap], #only load BandGap property
    #raw_path= './OMDB-GAP1_v1.1.tar.gz'
)

omdbdata.prepare_data()


pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=n_gaussians, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=n_interactions,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_gap = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=OMDB.BandGap)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_gap],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(OMDB.BandGap, add_mean=True, add_atomrefs=False)]
)

output_gap = spk.task.ModelOutput(
    name=OMDB.BandGap,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_gap],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": lr_rate}
)

logger = pl.loggers.TensorBoardLogger(save_dir=omdb_folder)

# Import profiler to debug the code
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
profiler = AdvancedProfiler(dirpath=".", filename="profiler_performance_logs")

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(omdb_folder, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

# profiler=profiler, # profiler, disabled for performance improvement
# logger=logger, # logger, disabled for performance improvement

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=False, # logger, disabled for performance improvement
    default_root_dir=omdb_folder,
    max_epochs=n_of_epochs,
    devices=1,
    accelerator="gpu",
    precision="32"
)

trainer.fit(task, datamodule=omdbdata)

# Learning time end
end_time = time.time()
print("Elapsed time in training:", end_time - start_time, "Seconds")

############################
######## Prediction ########

import torch
import numpy as np
from ase import Atoms

best_model = torch.load(os.path.join(omdb_folder, 'best_inference_model'), map_location=torch.device('cpu'))

for batch in omdbdata.test_dataloader():
    result = best_model(batch)
    print("Result dictionary:", result)
    break

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32)

numbers = np.array([6, 1, 1, 1, 1])
positions = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                      [0.002150416, -0.0060313176, 0.0019761204],
                      [1.0117308433, 1.4637511618, 0.0002765748],
                      [-0.540815069, 1.4475266138, -0.8766437152],
                      [-0.5238136345, 1.4379326443, 0.9063972942]])
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
print('Prediction:', atoms.get_total_energy())
