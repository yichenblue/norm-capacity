# The Shape of Generalization through the Lens of Norm-based Capacity Control

## Abstract
---

Understanding how the test risk scales with model complexity is a central question in machine learning. Classical theory is challenged by the learning curves observed for large over-parametrized deep networks.
Capacity measures based on parameter count typically fail to account for these empirical observations. To tackle this challenge, we consider norm-based capacity measures and develop our study for random features based estimators, widely used as simplified theoretical models for more complex networks. 
In this context, we provide a precise characterization of how the estimator’s norm concentrates and how it governs the associated test error. Our results show that the predicted learning curve admits a phase transition from under- to over-parameterization, but no double descent behavior. 
This confirms that more classical U-shaped behavior is recovered considering appropriate capacity measures based on models norms rather than size. 
From a technical point of view, we leverage deterministic equivalence as the key tool and further develop new deterministic quantities which are of independent interest.



![Introduction Figure](intro_figure.png?raw=true)

## Project Overview

This repository contains the implementation and analysis for our research on norm-based capacity control in machine learning. The project explores how model norms (rather than parameter counts) govern generalization behavior, with a focus on Random Feature Ridge Regression (RFRR) as a theoretical model for understanding deep networks.

## Project Structure

```
.
├── src/
│   ├── RFRR_real_world/           # Real-world dataset implementations
│   │   ├── main.py                # Main execution script
│   │   ├── models.py              # Model definitions
│   │   ├── model_utils.py         # Model utility functions
│   │   ├── utils.py               # General utility functions
│   │   ├── DE_utils.py            # Deterministic equivalent utilities
│   │   ├── visualize_intro_figure.py  # Visualization tools
│   │   ├── config.json            # Configuration file
│   │   ├── data/                  # Dataset storage
│   │   └── results/               # Experiment results
│   │
│   ├── Gaussian_design/           # Gaussian design implementations
│   │   ├── ridge_regression/      # Linear ridge regression
│   │   │   ├── ridge_regression.py
│   │   │   └── visualize.py
│   │   └── random_feature_ridge_regression/  # RFRR with Gaussian design
│   │       ├── random_feature_ridge_regression.py
│   │       └── visualize.py
│   │
│   └── Deep_NNs/                  # Deep neural network experiments
│       ├── deep_double_descent_ResNet18.py    # ResNet18 double descent
│       ├── mnist1d_deep_double_descent_cnn.py # CNN on MNIST-1D
│       ├── mnist1d_deep_double_descent_mlp.py # MLP on MNIST-1D
│       ├── saved_models_cnn/      # Pre-trained CNN models
│       ├── saved_models_mlp/      # Pre-trained MLP models
│       └── saved_models_ResNet/   # Pre-trained ResNet models
│
├── requirements.txt               # Project dependencies
└── README.md                     # This file
```

## Key Features

### 1. Random Feature Ridge Regression (RFRR)
- Implementation of RFRR for real-world datasets (MNIST)
- Deterministic equivalent analysis for theoretical insights
- Norm-based capacity control analysis

### 2. Gaussian Design Analysis
- Linear ridge regression with Gaussian design
- Random feature ridge regression with theoretical guarantees
- Visualization tools for learning curves

### 3. Deep Neural Network Experiments
- Double descent analysis on ResNet18
- MNIST-1D experiments with CNNs and MLPs
- Comprehensive model capacity studies

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Real-world Dataset Analysis

To run the RFRR analysis on MNIST dataset:

```bash
cd src/RFRR_real_world
python main.py --dataset MNIST --device cpu
```

### Gaussian Design Analysis

To run the Gaussian design analysis:

```bash
python src/gaussian_design/random_feature_ridge_regression/random_feature_ridge_regression.py
```

## Dependencies

The project requires the following main dependencies:
- PyTorch (>= 2.1.1)
- NumPy (>= 1.26.0)
- SciPy
- Matplotlib
- torchvision
- einops
- vit-pytorch

For a complete list of dependencies, see `requirements.txt`.

## Configuration

The project uses a `config.json` file to specify data and results directories. Make sure to create the necessary directories as specified in the configuration file.

## Results

The results of the experiments are saved in the `results/` directory, including:
- Test error vs features plots
- Norm vs features plots
- Test error vs Norm plots

## Contributing

Feel free to submit issues and enhancement requests.

