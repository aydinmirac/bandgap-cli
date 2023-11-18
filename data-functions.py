import os
import csv
import shutil

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

# Get the source and target directories
source_dir = './data-import/test'
destination_dir = './data-import/destination'

# CSV files
source_csv = './data-import/test/id_prop.csv'
destination_csv = './data-import/destination/id_prop.csv'

copy_files(source_dir, destination_dir)
edit_csv_files(source_csv, destination_csv)

