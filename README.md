# WIMN: Weather-Informed Mobility Network

This repository contains the implementation of WIMN (Weather-Informed Mobility Network), a deep learning model for inferring human mobility patterns under different weather conditions.

## Overview

WIMN is a novel deep learning architecture that combines Graph Neural Networks (GNN) with weather information to infer human mobility patterns after heavysnowfall. The model uses a hierarchical structure to capture mobility patterns at different scales, from hub nodes to top-level connections.

## Dataset

The dataset used for WIMN training, covering the Twin Cities Metropolitan Area. For convenience, the data has been pre-processed and saved as ".pt" files. After unzipping the data, you can directly start training WIMN using the command `python main.py`.

## Features

- Hierarchical mobility prediction using GNN
- Weather condition integration
- Multi-scale mobility pattern analysis
- Comprehensive evaluation metrics including:
  - Common Part of Commuters (CPC)
  - Pearson and Spearman correlations
  - Jensen-Shannon Distance
  - Root Mean Square Error (RMSE)
  - Normalized RMSE
  - Symmetric Mean Absolute Percentage Error (SMAPE)

## Project Structure

```
.
├── ablation study/     # Ablation study results and analysis
│   ├── inference results/    # Model inference outputs
│   ├── ground truth/        # Actual mobility patterns
│   └── deep edge weights/   # Deep edge weights for ROI 3
├── models/            # Model architecture implementations
│   └── model.py       # WIMN model implementation
├── Optim.py           # Optimization utilities
├── main.py           # Main training and evaluation script
└── data.zip          # Dataset (compressed)
```

## Requirements

The project requires the following Python packages:
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- SciPy
- scikit-learn

## Usage

1. Extract the data:
```bash
unzip data.zip
```

2. Run the training:
```bash
python main.py
```

## Model Architecture

WIMN consists of three main components:

1. **Encoder**: Processes input node and edge features using GNN layers
2. **Decoder**: Hierarchical decoder with three levels:
   - Top layer: Processes high-level mobility patterns
   - Hub layer: Handles hub node connections
   - Original layer: Processes detailed mobility patterns
3. **Edge-Node Fusion**: Combines edge and node features with weather information

## Evaluation Metrics

The model is evaluated using multiple metrics:
- CPC (Common Part of Commuters)
- Pearson Correlation
- Spearman Correlation
- Jensen-Shannon Distance
- RMSE (Root Mean Square Error)
- NRMSE (Normalized RMSE)
- SMAPE (Symmetric Mean Absolute Percentage Error)

## Ablation Study

The `ablation study` folder contains detailed analysis materials for ROI 3, including:
- Inference results from the WIMN model
- Ground truth data for comparison
- Deep edge weights extracted during forward propagation (obtained using PyTorch hooks)

These materials can be used to analyze the relationships between:
- Deep edge weights
- Snowfall depth
- Flow changes in the mobility patterns

This analysis helps understand how weather conditions influence mobility patterns at different scales of the network.

