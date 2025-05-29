# CNN Implementation from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch for image classification using the CIFAR-10 dataset. The implementation includes various architectural variations to study the impact of different hyperparameters.

## Project Structure

```
.
├── README.md
├── layers.py        # Implementation of CNN layers (Conv2D, ReLU, MaxPool2D, etc.)
├── models.py        # Different CNN architectures
├── train.py        # Training utilities and loss functions
├── utils.py        # Data loading and preprocessing
├── experiment.py   # Experiment runner
└── main.py         # Main entry point
```

## Features

1. **Layer Implementations**:
   - Convolutional Layer (Conv2D)
   - ReLU Activation
   - Max Pooling
   - Average Pooling
   - Flatten Layer
   - Dense (Fully Connected) Layer

2. **Model Variations**:
   - Different numbers of convolutional layers (2, 3, 4)
   - Different filter counts per layer
   - Different kernel sizes (3x3, 5x5)
   - Different pooling types (Max vs Average)

3. **Training**:
   - Sparse Categorical Crossentropy Loss
   - Adam Optimizer
   - Training/Validation Split (40k/10k)

4. **Evaluation**:
   - Macro F1-Score
   - Loss Curves Visualization

## Running Experiments

To run all experiments:

```bash
python main.py
```

This will:
1. Train different model variations
2. Generate loss plots in the `plots` directory
3. Print F1-scores for each model variation

## Results

The experiments analyze the impact of:
1. Number of convolutional layers
2. Number of filters per layer
3. Kernel sizes
4. Pooling types

Results are saved as plots in the `plots` directory and printed to console. 