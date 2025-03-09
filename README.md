# ColaGNN: Cross-location Attention based Graph Neural Networks

This repository contains the implementation of several models for spatiotemporal forecasting, including the ColaGNN model from the paper [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf) (CIKM 2020).

## Overview

The project implements multiple spatiotemporal forecasting models that can predict time series data across multiple locations by capturing both temporal dependencies and spatial relationships between locations.

### Implemented Models

- **ColaGNN**: Cross-location Attention based Graph Neural Network that combines RNN for temporal modeling and GNN for spatial modeling with a cross-location attention mechanism
- **SelfAttnRNN**: Self-attention enhanced RNN for temporal modeling
- **ST-GAT**: Spatiotemporal Transformer with Graph Attention Networks
- Traditional models: **AR**, **ARMA**, **VAR**, **GAR**
- Deep learning models: **RNN**, **CNNRNN_Res**, **LSTNet**, **STGCN**, **DCRNN**

## Datasets

The datasets are in the `data` folder. Each dataset consists of two files:
- **Time series data**: (e.g., `japan.txt`) - Contains spatiotemporal data with columns representing locations (e.g., prefectures) and rows representing timestamps (e.g., weeks).
- **Adjacency matrix**: (e.g., `japan-adj.txt`) - Contains spatial relationship information between locations.

Available datasets:
- **Japan prefectures**: `japan.txt` and `japan-adj.txt`
- **US regions**: `region785.txt` and `region-adj.txt`
- **US states**: `state360.txt` and `state-adj.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/colagnn.git
cd colagnn

# Create a virtual environment (optional)
conda create -n colagnn_env python=3.8
conda activate colagnn_env

# Install dependencies
pip install torch torchvision
pip install torch-geometric  # For GAT implementation
pip install tensorboardX
pip install scikit-learn
pip install scipy numpy
```

## Usage

### Training a Model

Example commands for training various models:

```bash
# Train ColaGNN model on Japan dataset
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model cola_gnn --patience 100 --gpu 0 --mylog

# Train SelfAttnRNN model on Japan dataset this takes longer to run
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model SelfAttnRNN --patience 100 --gpu 0 --mylog

# Train ST-GAT model on Japan dataset
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model st_gat --patience 100 --gpu 0 --mylog

# Train CNNRNN_Res model on Japan dataset
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model cnnrnn_res --patience 100 --gpu 0 --mylog

# Train LSTNet model on Japan dataset
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model lstnet --patience 100 --gpu 0 --mylog

# Train STGCN model on Japan dataset
python src/train.py --dataset japan --sim_mat japan-adj --epoch 1500 --train 0.5 --val 0.2 --test 0.3 --batch 128 --horizon 5 --model stgcn --patience 100 --gpu 0 --mylog
```

### Key Parameters

- `--dataset`: Name of the dataset (e.g., japan, region, state)
- `--sim_mat`: Name of the adjacency matrix file (e.g., japan-adj)
- `--model`: Model architecture to use (see list of implemented models)
- `--window`: Size of historical window for prediction (default: 20)
- `--horizon`: Number of future time steps to predict (default: 1)
- `--train`/`--val`/`--test`: Ratios for splitting data
- `--batch`: Batch size for training
- `--n_hidden`: Hidden dimension size for RNN/GNN layers
- `--patience`: Early stopping patience
- `--cuda`/`--gpu`: GPU settings

## Model Architecture

### ColaGNN

ColaGNN captures both temporal and spatial dependencies by:
1. Using RNN for temporal feature extraction
2. Employing a novel cross-location attention mechanism to model dynamic spatial dependencies
3. Applying graph convolutional layers for spatial feature extraction
4. Combining spatial and temporal features for the final prediction

### SelfAttnRNN

Enhanced RNN architecture with self-attention for improved temporal modeling and multi-horizon forecasting.

### ST-GAT

Combines transformer architecture for temporal dependencies with Graph Attention Networks for spatial modeling.

## Training Data

The **DataBasicLoader** class in `src/data.py` handles data processing:
- `_split`: Splits data into training, validation, and test sets
- `_batchify`: Generates data samples, each containing a time series input with length equal to `window` and prediction targets with length equal to `horizon`
- `get_batches`: Generates mini-batches for training

## Results

Model performance is evaluated using several metrics:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- PCC: Pearson Correlation Coefficient
- R²: Coefficient of Determination
- Peak MAE: MAE during peak periods

## TensorBoard Visualization

You can visualize training progress using TensorBoard:

```bash
tensorboard --logdir=tensorboard/
```

## Citation

If you use this code, please cite the original paper:

```
@inproceedings{deng2020cola,
  title={Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction},
  author={Deng, Songgaojun and Wang, Shusen and Rangwala, Huzefa and Wang, Lijing and Ning, Yue},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={245--254},
  year={2020}
}
```

## License

[MIT License](LICENSE)
