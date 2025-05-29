from utils import load_cifar10
from models import (
    CNN2Layer, CNN3Layer, CNN4Layer,
    CNNWideFilters, CNNLargeKernel, CNNAvgPool
)
from train import train, evaluate
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}.png')
    plt.close()

def run_experiment(model_class, x_train, y_train, x_val, y_val, title):
    print(f"\nRunning experiment: {title}")
    model = model_class()
    train_losses, val_losses = train(model, x_train, y_train, x_val, y_val, epochs=10)
    f1 = evaluate(model, x_val, y_val)
    plot_losses(train_losses, val_losses, title)
    return train_losses, val_losses, f1

def main():
    # Load data
    x_train, y_train, x_val, y_val = load_cifar10()
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Experiment 1: Number of Convolutional Layers
    models = [
        (CNN2Layer, "2 Conv Layers"),
        (CNN3Layer, "3 Conv Layers"),
        (CNN4Layer, "4 Conv Layers")
    ]
    
    results = []
    for model_class, name in models:
        _, _, f1 = run_experiment(model_class, x_train, y_train, x_val, y_val, name)
        results.append((name, f1))
    
    print("\nResults - Number of Conv Layers:")
    for name, f1 in results:
        print(f"{name}: F1 Score = {f1:.4f}")
    
    # Experiment 2: Filter Counts
    models = [
        (CNN2Layer, "Base Filters (8, 16)"),
        (CNNWideFilters, "Wide Filters (32, 64)")
    ]
    
    results = []
    for model_class, name in models:
        _, _, f1 = run_experiment(model_class, x_train, y_train, x_val, y_val, name)
        results.append((name, f1))
    
    print("\nResults - Filter Counts:")
    for name, f1 in results:
        print(f"{name}: F1 Score = {f1:.4f}")
    
    # Experiment 3: Kernel Sizes
    models = [
        (CNN2Layer, "3x3 Kernels"),
        (CNNLargeKernel, "5x5 Kernels")
    ]
    
    results = []
    for model_class, name in models:
        _, _, f1 = run_experiment(model_class, x_train, y_train, x_val, y_val, name)
        results.append((name, f1))
    
    print("\nResults - Kernel Sizes:")
    for name, f1 in results:
        print(f"{name}: F1 Score = {f1:.4f}")
    
    # Experiment 4: Pooling Types
    models = [
        (CNN2Layer, "Max Pooling"),
        (CNNAvgPool, "Average Pooling")
    ]
    
    results = []
    for model_class, name in models:
        _, _, f1 = run_experiment(model_class, x_train, y_train, x_val, y_val, name)
        results.append((name, f1))
    
    print("\nResults - Pooling Types:")
    for name, f1 in results:
        print(f"{name}: F1 Score = {f1:.4f}")

if __name__ == "__main__":
    main() 