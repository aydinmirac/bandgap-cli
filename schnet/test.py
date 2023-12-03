import os
import sys
import schnetpack as spk
import schnetpack.transform as trn
import torch
import ase.io
import datetime

# Parameters from CLI arguments
model_folder = sys.argv[1]
molecule_file = sys.argv[2]
cutoff = float(sys.argv[3])

def test_schnet(model_folder, molecule_file, cutoff):

    # Load the best model
    best_model = torch.load(os.path.join(model_folder, 'best_inference_model'), map_location=torch.device('cpu'))

    # Create a converter for test structure
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=cutoff), dtype=torch.float32)

    # Read the CIF file
    atoms = ase.io.read(molecule_file)

    # Convert the atom and predict the gap
    inputs = converter(atoms)
    pred = best_model(inputs)

    # Write the required output to a file
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"./predictions/schnet_result_{timestamp}.txt"
    #print("Number of atoms in the file:", len(atoms))
    #print('Prediction:', pred["band_gap"].item(), "eV")
    with open(filename, 'w') as f:
        f.write("The number of atoms in the file: {}\n".format(len(atoms)))
        f.write("Prediction: {} eV\n".format(pred["band_gap"].item()))

    f.close()

# Call the function
test_schnet(model_folder, molecule_file, cutoff)
