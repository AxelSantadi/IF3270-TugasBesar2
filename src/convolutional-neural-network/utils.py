import numpy as np
from tensorflow.keras.datasets import cifar10

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val = x_train[:40000], x_train[40000:]
    y_train, y_val = y_train[:40000], y_train[40000:]
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    
    # NCHW format: (batch, channels, height, width)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_val = np.transpose(x_val, (0, 3, 1, 2))

    return x_train, y_train, x_val, y_val
