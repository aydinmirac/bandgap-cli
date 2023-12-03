import os
import sys
import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase as OMDB
import schnetpack.transform as trn
import torch
import torchmetrics
import pytorch_lightning as pl
from ase.db import connect
import numpy as np
import time


# Parameters from CLI arguments
batchsize = int(sys.argv[1])
cutoff = float(sys.argv[2])
num_of_workers = int(sys.argv[3])
n_atom_basis = int(sys.argv[4])
n_gaussians = int(sys.argv[5])
n_interactions = int(sys.argv[6])
lr_rate = float(sys.argv[7])
n_of_epochs = int(sys.argv[8])
train_size = int(sys.argv[9])
val_size = int(sys.argv[10])
test_size = int(sys.argv[11])

def train_schnet(batchsize, cutoff, num_of_workers, n_atom_basis, n_gaussians, n_interactions, lr_rate, n_of_epochs, train_size, val_size, test_size):
    # Create an incremental folder name to store training results from different runs
    base_folder = "results"
    existing_folders = [folder for folder in os.listdir("./schnet") if folder.startswith(base_folder)]

    if not existing_folders:
        # If no folders exist, create the first one
        new_folder = f"{base_folder}_1"
    else:
        # If folders exist, find the latest one and increment the number
        latest_folder = max(existing_folders, key=lambda folder: int(folder.split('_')[-1]))
        latest_number = int(latest_folder.split('_')[-1])
        new_folder = f"{base_folder}_{latest_number + 1}"

    # Create the new folder
    os.makedirs(os.path.join("./schnet", new_folder))
    print(f"New folder created: {new_folder}")

    result_folder = os.path.join("./schnet", new_folder)
    
    # Print Hyperparameters 
    print("#################")
    print("HYPERPARAMETERS")
    print("#################")
    print("Batch size:", batchsize)
    print("Cutoff Ratio:", cutoff)
    print("Number of Workers:", num_of_workers)
    print("Number of Atom Features:", n_atom_basis)
    print("Number of Gaussian Functions:", n_gaussians)
    print("Number of Interactions:", n_interactions)
    print("Learning Rate", lr_rate)
    print("Number of Epoch:", n_of_epochs)
    print("Training Size:", train_size)
    print("Validation Size:", val_size)
    print("Test Size:", test_size)

    # Add timer
    start_time = time.time()

    # Increase Torch precision
    torch.set_float32_matmul_precision("medium")

    omdbdata = OMDB(
        datapath="./schnet/all_molecules.db",
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
        split_file=os.path.join(result_folder, "split.npz"),
        pin_memory=True, # set to false, when not using a GPU
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

    logger = pl.loggers.TensorBoardLogger(save_dir=result_folder)

    # Import profiler to debug the code
    #from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
    #profiler = AdvancedProfiler(dirpath=".", filename="profiler_performance_logs")

    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(result_folder, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    # profiler=profiler, # profiler, disabled for performance improvement
    # logger=logger, # logger, disabled for performance improvement

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=False, # logger, disabled for performance improvement
        default_root_dir=result_folder,
        max_epochs=n_of_epochs,
        devices=1,
        accelerator="gpu",
    )

    trainer.fit(task, datamodule=omdbdata)

    # Learning time end
    end_time = time.time()
    print("Elapsed time in training:", end_time - start_time, "Seconds")

# Call the function
train_schnet(batchsize, cutoff, num_of_workers, n_atom_basis, n_gaussians, n_interactions, lr_rate, n_of_epochs, train_size, val_size, test_size)
