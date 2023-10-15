from ase.io import read, write
import numpy as np
import os
from tqdm import tqdm
import glob

# Extract OMDB database from scratch if necessary
def extract_omdb():

    # Create folder to store cif files
    #os.makedirs("./omdb_cifs", exist_ok=True)

    # Change directory to datasets/omdb
    os.chdir('datasets/omdb')

    # Read the structure file including all molecules
    materials = read('structures.xyz', index=':')

    # Read the bandgap information
    bandgaps = np.loadtxt('bandgaps.csv', dtype=float)

    # Read the COD database file
    cods = np.loadtxt('CODids.csv', dtype=int)

    
    # Convert structures to cif files
    for idx in tqdm(range(len(materials))):
        write(f"{cods[idx]}.cif", materials[idx])

    # Save COD ID and bandgap information to csv file
    np.savetxt("id_prop.csv",
            np.array(list(zip(cods.astype(float), bandgaps.astype(float)))),
            delimiter=',', fmt=["%i", "%f"])
    