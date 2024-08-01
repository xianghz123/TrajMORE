# TrajMORE Training Script

## Overview
This script is designed for training the TrajMORE model on trajectory datasets. The TrajMORE model is intended for metric learning tasks involving trajectory data, capable of learning embeddings that can be used for trajectory similarity assessment, clustering, or classification.

## Dependencies
- Python 3.x
- PyTorch
- Numpy
- Visdom (optional for visualization)
- Other dependencies as required by the grid2vec and TrajMORE modules, as well as utility scripts provided (utils.py, etc.).

## Installation
1. Ensure you have Python 3.x installed on your system. It's recommended to use a virtual environment:
    ```sh
    python3 -m venv TrajMORE-env
    source TrajMORE-env/bin/activate
    ```

2. Install the required Python packages:
    ```sh
    pip install torch numpy visdom argparse
    ```

## Preparing Your Dataset
You need to prepare your trajectory dataset in the expected format. The dataset should include:
- `trajs`: List of trajectory indices.
- `origin_trajs`: List of original trajectories with (lon, lat) coordinates.
- `dis_matrix`: Distance matrix of trajectories.

Make sure you have separate files for training, validation, and testing datasets.

## Running the Script
To train the TrajMORE model, you need to prepare your trajectory dataset and adjust parameters according to your dataset characteristics.

1. **Prepare your trajectory dataset** in the required format, ensuring you have training, validation, and test sets ready.
2. **Adjust parameters** in the `parameters.py` script or prepare to override them via command-line arguments when running the training script.

### Example Command
Run the script with the desired parameters:
```sh
python train.py --batch_size 32 --epoch_num 100 --learning_rate 0.001 --device cuda
