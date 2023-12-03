# CLI tool for bandgap prediction
This repository includes an automation for the following Deep Learning packages that are used to create models for atomistic systems:
1. [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
2. [ALIGNN](https://github.com/usnistgov/alignn)

## 1. Requirements
1. Python 3.10+
2. CUDA 11.8
3. [Typer](https://typer.tiangolo.com/)
4. Linux or MacOS
5. Bash Shell (If you are using MacOS, you can change your shell or change the Shebang line in every bash script in the repository to `#!/bin/zsh`)

## 2. Limitations
1. Since PyTorch officially supports Apple silicon GPU acceleration, sometimes it throws error in previous versions:

   `Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.`

   Therefore, it is better to utilize Linux system instead of MacOS. But users are free to implement their own solution in this repository.

2. Currently, the tool only support CIF file format. If you are using other format such as XYZ, you can convert it with [VESTA](https://jp-minerals.org/vesta/en/) to CIF format. You can draw a unit cell around molecule and export it as a CIF file. We will support other file formats in the future.

## Workflow
After downloading the repository, you can follow these steps to use the tool:
1. Install SchNetPack and ALIGNN with `install-schnet` and `install-alignn` commands respectively.
2. Extract the dataset with `extract-dataset` command.
3. Create a database for SchNetPack with `create-database` command.

Now you are ready to train the models. You can find more information how to use command line options below.

## 3. Configuration Steps
### 3.1. Installation of Typer
After having Python in your system, you can install Typer library for CLI commands.

```
pip install "typer[all]"
```

It is recommended that you can create a virtual environment with Python or Conda environment to install requirements. For Python virtual environment, you can follow these steps:

```
pip install virtualenv
python -m venv <environment-name>
source <environment-name>/bin/activate
pip install "typer[all]"
```
With these steps, you can maintain additional python packages in your own directories.

### 3.2. Cloning the repository
You can simply run the following command to download the repository:

```
git clone https://github.com/aydinmirac/bandgap-prediction.git
```
## 4. Commands
All of commands and their explanations can be seen by typing `--help` option:

```
host$ python main.py --help
                                                                                                                                                                                                            
 Usage: main.py [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                 
                                                                                                                                                                                                            
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                                                  │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                           │
│ --help                        Show this message and exit.                                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ copy-data                       This command copies your molecules to SchNetPack and ALIGNN data directories.                                                                                            │
│ create-database                 This command creates a ASE database for SchNetPack model.                                                                                                                │
│ extract-dataset                 This command extracts necessary dataset for both ALIGNN and SchNetPack.                                                                                                  │
│ install-alignn                  This command installs ALIGNN package.                                                                                                                                    │
│ install-schnet                  This command installs SchNetPack package.                                                                                                                                │
│ test-alignn                     This command predicts the bandgap of a single molecule with the trained ALIGNN model.                                                                                    │
│ test-schnet                     This command predicts the bandgap of a single molecule with the trained SchNetPack model.                                                                                │
│ train-alignn                    This command starts a training process by using ALIGNN model.                                                                                                            │
│ train-schnet                    This command starts a training process by using SchNetPack model.                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

If you want to see the command options, you can simply use command name and `--help` option again:


```
host$ python main.py train-alignn --help
                                                                                                                                                                                                            
 Usage: main.py train-alignn [OPTIONS] [N_VAL] [N_TEST] [N_TRAIN] [EPOCHS]                                                                                                                                  
                             [BATCH_SIZE] [LEARNING_RATE] [NUM_WORKERS]                                                                                                                                     
                             [CUTOFF] [MAX_NEIGHBORS]                                                                                                                                                       
                                                                                                                                                                                                            
 This command starts a training process by using ALIGNN model.                                                                                                                                              
                                                                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   n_val              [N_VAL]          The size of validation set [default: 100]                                                                                                                          │
│   n_test             [N_TEST]         The size of test set [default: 100]                                                                                                                                │
│   n_train            [N_TRAIN]        The size of train set [default: 100]                                                                                                                               │
│   epochs             [EPOCHS]         The epoch size [default: 10]                                                                                                                                       │
│   batch_size         [BATCH_SIZE]     The batch size [default: 20]                                                                                                                                       │
│   learning_rate      [LEARNING_RATE]  learning rate in float [default: 0.01]                                                                                                                             │
│   num_workers        [NUM_WORKERS]    The size of validation set [default: 4]                                                                                                                            │
│   cutoff             [CUTOFF]         The cutoff ratio. Please check Schetpack or Alignn document for it [default: 5.0]                                                                                  │
│   max_neighbors      [MAX_NEIGHBORS]  The maximum number of neighbours. Please check Schetpack or Alignn document for it [default: 8]                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 5. Usage of CLI Options
### 5.1. `install-alignn`
This command installs ALIGNN package and other required libraries. You can run the following command to install ALIGNN:

```
python main.py install-alignn
```
It will create a Python virtual environment and install necessary packages.

### 5.2. `install-schnet`
This command installs SchNetPack package and other required libraries. You can run the following command to install SchNetPack:

```
python main.py install-schnet
```
It will create a Python virtual environment and install necessary packages.

### 5.3. `create-database`
SchNetPack requires an ASE compatible database to train and test molecules. This command will create a ASE database for SchNetPack. You can run the following command:

```
python main.py create-database
```

This will create the database under `schnet` directory.

### 5.4. `extract-dataset`
The dataset in this repository is based on [OMDB](https://omdb.mathub.io/dataset) dataset. Since Deep Learning algorithms require lots of data, this dataset has been augmented with [AugLiChem](https://github.com/BaratiLab/AugLiChem) to increase the number of molecules and other structures from other researches have been added to improve the prediction quality. 

The command will extract all CIF structures and other necessary files such as `id_prop.csv` which include CIF file name and corresponding bandgap value and `atom_init.json`. You can run the following command:

```
python main.py extract-dataset
```
### 5.5 `copy-data`
If you want to add more CIF mocule data to the current dataset, you can use this command. it will copy your molecules from `data_copy` folder to `datasets` folder and edit `id_prop.csv` file in the target folder. But for this operation, there are some rules:

1. You should provide the CIF file/s and `id_prop.csv` file under `data_copy` folder.
2. Your CSV file must include the file name/s and its bangdap value. The content of your CSV file must be like this:

    ```
    1008775.cif,2.456451
    1008776.cif,3.498481
    1008787.cif,2.767671
    1100226.cif,4.223321
    1100244.cif,3.465411
    ```

In order to copy to your files to the `datasets` folder, you can run the following command:

```
python main.py copy-data
```

### 5.6 `train-schnet`
In order to train SchNetPack model, you have to give some parameters. These parameters are shown below:

```
host$ python main.py train-schnet --help
                                                                                                                                                                                                                            
 Usage: main.py train-schnet [OPTIONS] [BATCHSIZE] [CUTOFF] [NUM_OF_WORKERS]                                                                                                                                                
                             [N_ATOM_BASIS] [N_GAUSSIANS] [N_INTERACTIONS]                                                                                                                                                  
                             [LR_RATE] [N_OF_EPOCHS] [TRAIN_SIZE] [VAL_SIZE]                                                                                                                                                
                             [TEST_SIZE]                                                                                                                                                                                    
                                                                                                                                                                                                                            
 This command starts a training process by using SchNetPack model.                                                                                                                                                          
                                                                                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   batchsize           [BATCHSIZE]       The batch size [default: 32]                                                                                                                                                     │
│   cutoff              [CUTOFF]          The cutoff ratio [default: 5.0]                                                                                                                                                  │
│   num_of_workers      [NUM_OF_WORKERS]  The number of workers for data load [default: 4]                                                                                                                                 │
│   n_atom_basis        [N_ATOM_BASIS]    The number of features to describe atomic environments [default: 128]                                                                                                            │
│   n_gaussians         [N_GAUSSIANS]     The number of Gaussian fuctions [default: 50]                                                                                                                                    │
│   n_interactions      [N_INTERACTIONS]  The number of interaction blocks [default: 3]                                                                                                                                    │
│   lr_rate             [LR_RATE]         The learning rate [default: 0.01]                                                                                                                                                │
│   n_of_epochs         [N_OF_EPOCHS]     The epoch size [default: 5]                                                                                                                                                      │
│   train_size          [TRAIN_SIZE]      The size of training set [default: 1000]                                                                                                                                         │
│   val_size            [VAL_SIZE]        The size of validation set [default: 100]                                                                                                                                        │
│   test_size           [TEST_SIZE]       The size of test set [default: 100]                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
You have to give all parameters in the order that was shown above. For example:

```
python main.py train-schnet 40 8.0 8 64 30 3 0.001 10 2000 200 200
```
This will assign parameters accoding to the command line options.
```
batchsize=40
cutoff=8.0
num_of_workers=8
n_atom_basis=64
n_gaussians=30
n_interactions=3
lr_rate=0.001
n_of_epochs=10
train_size=2000
val_size=200
test_size=200
```
After the training process is done, you will see your result folder under `schnet`.

### 5.7 `test-schnet`
If you want to test a molecule, you can use this command. This command will take your CIF molecule file and convert it into ASE database format. Then it will predict the band gap of that molecule. In order to use this command, please copy your file under `predictions` folder. The command will read your file from there.

```
host$ python main.py test-schnet --help                                 
                                                                                                                                                                                                                            
 Usage: main.py test-schnet [OPTIONS] [MODEL_FOLDER] [MOLECULE_FILE] [CUTOFF]                                                                                                                                               
                                                                                                                                                                                                                            
 This command predicts the bandgap of a single molecule with the trained SchNetPack model.                                                                                                                                  
                                                                                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   model_folder       [MODEL_FOLDER]   The folder name that includes 'best_inference_model' from your last run [default: ./schnet/results_1]                                                                              │
│   molecule_file      [MOLECULE_FILE]  The CIF file that you want to predict [default: ./predictions/1008775.cif]                                                                                                         │
│   cutoff             [CUTOFF]         The cutoff ratio [default: 5.0]                                                                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
An example for how to use this command is given below:

```
python main.py test-schnet ./schnet/results_1 ./predictions/1008775.cif 5.0
```
The command will take the trained model under `results_1` folder and predict the band gap of `1008775.cif` file. Your results will be written to a file under `predictions` folder.

### 5.8 `train-alignn`
If you want to use ALIGNN model for training, you can run this command.

```
python main.py train-alignn --help                                          
                                                                                                                                                                                                                            
 Usage: main.py train-alignn [OPTIONS] [N_VAL] [N_TEST] [N_TRAIN] [EPOCHS]                                                                                                                                                  
                             [BATCH_SIZE] [LEARNING_RATE] [NUM_WORKERS]                                                                                                                                                     
                             [CUTOFF] [MAX_NEIGHBORS]                                                                                                                                                                       
                                                                                                                                                                                                                            
 This command starts a training process by using ALIGNN model.                                                                                                                                                              
                                                                                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   n_val              [N_VAL]          The size of validation set [default: 100]                                                                                                                                          │
│   n_test             [N_TEST]         The size of test set [default: 100]                                                                                                                                                │
│   n_train            [N_TRAIN]        The size of train set [default: 100]                                                                                                                                               │
│   epochs             [EPOCHS]         The epoch size [default: 10]                                                                                                                                                       │
│   batch_size         [BATCH_SIZE]     The batch size [default: 20]                                                                                                                                                       │
│   learning_rate      [LEARNING_RATE]  learning rate in float [default: 0.01]                                                                                                                                             │
│   num_workers        [NUM_WORKERS]    The number of workers for data load [default: 4]                                                                                                                                   │
│   cutoff             [CUTOFF]         The cutoff ratio. Please check Schetpack or Alignn document for it [default: 5.0]                                                                                                  │
│   max_neighbors      [MAX_NEIGHBORS]  The maximum number of neighbours. Please check Schetpack or Alignn document for it [default: 8]                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
An example is given below:

```
python main.py train-alignn 100 100 1000 20 40 0.00001 16 6.0 12
```
According to this values, the following parameters will be assigned to the model:

```
n_val=100
n_test=100
n_train=1000
epochs=20
batch_size=40
learning_rate=0.00001
num_workers=16
cutoff=6.0
max_neighbors=12
```
These parameters will be passed into `config.json` under `alignn` folder. The config file will be used to train ALIGNN model. The result of the training will be created under `alignn/results_$date` folder.

### 5.9 `test-alignn`
After training your ALIGNN model, you can test a structure. Like in SchNetPack, your CIF file must be under `predictions` folder.

```
host$ python main.py test-alignn --help                               
                                                                                                                                                                                                                            
 Usage: main.py test-alignn [OPTIONS] BEST_MODEL [FILE_FORMAT] [MOLECULE_PATH]                                                                                                                                              
                            [CUTOFF]                                                                                                                                                                                        
                                                                                                                                                                                                                            
 This command predicts the bandgap of a single molecule with the trained ALIGNN model.                                                                                                                                      
                                                                                                                                                                                                                            
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    best_model         TEXT             The path of best model from your last training, stored in the last output folder as best_model.pt [default: None] [required]                                                    │
│      file_format        [FILE_FORMAT]    The file format, currently only cif supported [default: cif]                                                                                                                    │
│      molecule_path      [MOLECULE_PATH]  The path of your test molecule, please store it under 'predictions' folder [default: ./predictions/1008775.cif]                                                                 │
│      cutoff             [CUTOFF]         The size of validation set [default: 5.0]                                                                                                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

An example for testing is given below:
```
python main.py test-alignn ./alignn/results_2023.01.01:20.21.22/best_model.pt cif ./predictions/1008775.cif 5.0
```
The command will read your best model to predict the band gap of `1008775.cif` file. The results will be written to file under `predictions` folder.
