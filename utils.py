import os
import csv
import shutil
import json

# Copy molecules from data_copy to datasets folder
def copy_files(source_dir, destination_dir):
    # Check if the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy CIF files
    for filename in os.listdir(source_dir):
        if filename.endswith('.cif'):
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename)

            if not os.path.exists(destination_file):
                print(f"Copying file: {filename}")
                shutil.copy2(source_file, destination_file)
            else:
                print(f"File already exists in the destination directory: {filename}")

# Edit CSVs in dataset folders
def edit_csv_files(source_csv, destination_csv):
    # Read the source CSV file
    with open(source_csv, 'r') as source_file:
        source_reader = csv.reader(source_file)
        source_data = list(source_reader)

    # Check if the destination CSV file exists
    if not os.path.exists(destination_csv):
        with open(destination_csv, 'w') as destination_file:
            destination_writer = csv.writer(destination_file)
            destination_writer.writerow(['filename', 'bandgap'])

            for row in source_data:
                destination_writer.writerow(row)
    else:
        # Read the destination CSV file
        with open(destination_csv, 'r') as destination_file:
            destination_reader = csv.reader(destination_file)
            destination_data = list(destination_reader)

        # Check if the data from the source CSV file already exists in the destination CSV file
        for row in source_data:
            if row in destination_data:
                print(f"Data already exists in the destination CSV file: {row}")
            else:
                # Append the data from the source CSV file to the destination CSV file
                with open(destination_csv, 'a') as destination_file:
                    destination_writer = csv.writer(destination_file)
                    destination_writer.writerow(row)

def edit_json_file(n_val, n_test, n_train, epochs, batch_size, learning_rate, num_workers, cutoff, max_neighbors):

    # Open json file for ALIGNN
    with open('./alignn/config.json', 'r') as f:
        config_data = json.load(f)

    # Update the key-value pairs with the provided arguments
    config_data['n_val'] = n_val
    config_data['n_test'] = n_test
    config_data['n_train'] = n_train
    config_data['epochs'] = epochs
    config_data['batch_size'] = batch_size
    config_data['learning_rate'] = learning_rate
    config_data['num_workers'] = num_workers
    config_data['cutoff'] = cutoff
    config_data['max_neighbors'] = max_neighbors

    # Open the JSON file in write mode and dump the updated data
    with open('./alignn/config.json', 'w') as f:
        json.dump(config_data, f, indent=4)