# Project Name

## Overview
This project involves the creation and training of machine learning models for 3D object recognition. The project is structured into different directories for dataset creation and model training. The models used in this project include the HEU model and the CMT model, which can be downloaded from the provided links.

## Code Structure

### 1. `datacreation` Directory
This directory is responsible for the creation of datasets used in training the models. It includes scripts and utilities for:

- **Main2** generates multi-views normal and abnormal data.
- **utils** include helper functions to deal with some complex task


### 2. `looking3D` Directory
This directory contains the main scripts and data for training the 3D models. It includes:

- **Model training:** The main script to train the HEU and CMT models.
-

#### Key Files and Folders:
- `train.py`: The main script for training both the HEU and CMT models. It handles loading the dataset, configuring the model, and managing the training process. It also handles 
- `evaluate.py`: The script for the evaluation for both CMT and HEU models.
- `/data/dataset_e2e.py`: This script is used to load the dataset using a heuristic approach if needed. It provides flexibility in how the data is prepared and fed into the training process.

To start the training process, run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port ${MASTER_PORT} train.py --exp_name CMT-final --data_path ${DATA_PATH} --epochs 8 --lr 2e-5 --batch_size 8 --num_workers 4 --pred_box --num_mesh_images 20
```
To start the training process, run the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port ${MASTER_PORT} evaluate.py --data_path  ${DATA_PATH} --resume_ckpt experiments/CMT-final/checkpoints/${MODEL} --num_mesh_images 20  --batch_size 8 --num_workers 4 --pred_box 
```
### 3. Models
The trained models can be downloaded using the following links:

- **HEU Model**: [Download here](https://drive.google.com/file/d/1Bq1AHfasap9THLUqAINBzwM9RDT-IJSW/view?usp=drive_link)
- **CMT Model**: [Download here](https://drive.google.com/file/d/1zf6JtegE3qajwln-mgnJM8_4_WK2k3DM/view?usp=drive_link)

Place the downloaded models in the `models/` directory within the `looking3D` folder before running any evaluation scripts.

### 4. Dataset
The dataset can be found using the dollowing link:
- **Dataset**: [Download here](https://drive.google.com/file/d/1BH9mS2uoiezk5zXMj0RKAeIxJ393FHJB/view?usp=sharing)


