#from ase.io import read, write
#import numpy as np
#import os
#from tqdm import tqdm
#import glob

# Bandgap manipulation and data appending
import csv
import os
import random


###############################
# *************************** #
#  DATA AND CSV MANIPULATION  #
###############################
# *************************** #

# Keywords to identify files for band gap adjustment
keywords = ['perturbed', 'rotated', 'translate', 'swapaxes']

# Randomize bandgap assignment after augmentation
def adjust_bandgap(bandgap):
    new_gap = str(round(random.uniform(float(bandgap) - 0.20, float(bandgap) + 0.20),6))
    return new_gap

# Manipulate the CSV files and create a new one
def process_csv(input_file, output_file):
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    
    modified_data = []

    for row in data:
        filename, bandgap = row
        modified_bandgap = bandgap

        for keyword in keywords:
            if keyword in filename:
                modified_bandgap = adjust_bandgap(bandgap)
        
        modified_data.append([filename, modified_bandgap])
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(modified_data)

