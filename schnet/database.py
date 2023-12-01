import time
from ase.io import read, iread
import numpy as np
import csv
import os
from schnetpack.data import ASEAtomsData
from tqdm import tqdm

# Main code
start = time.time()

cif_dir = "./datasets/main"
csv_file = "./datasets/main/id_prop.csv"

# Define a function to read the CSV file containing molecule IDs and bandgaps
def read_bandgap_data():
    with open(csv_file) as file:
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
    return rows

# Define a function to create a list of CIF
def get_cif_file_list(directory):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(".cif"):
            file_list.append(file)

    return file_list

# Define a function to match CIF files with corresponding bandgaps and create ASE Atoms objects
def create_ase_atoms_and_properties(directory):
    atoms = []
    property_list = []

    bandgap_data = read_bandgap_data()
    cif_file_list = get_cif_file_list(directory)
    pbar = tqdm(total=len(bandgap_data))
    for gap in bandgap_data:
        pbar.update(1)
        for name in cif_file_list:
            if name == gap[0]:
                file_relative_path = os.path.join(directory, name)
                atoms.append(read(file_relative_path))
                bandgap = np.array([float(gap[1])], dtype=np.float32)
                property_list.append({'band_gap': bandgap})

    pbar.close()
    return atoms, property_list


atoms, property_list = create_ase_atoms_and_properties(cif_dir)

end = time.time()
print("Time elapsed:", end - start, "seconds")

print("Length of atoms:", len(atoms), "and gap values:", len(property_list))

new_dataset = ASEAtomsData.create('./schnet/all_molecules.db', distance_unit='Ang', property_unit_dict={'band_gap':'eV'})
new_dataset.add_systems(property_list, atoms)

print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

for p in new_dataset.available_properties:
    print('-', p)
