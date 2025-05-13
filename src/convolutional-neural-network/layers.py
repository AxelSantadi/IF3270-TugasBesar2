import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / in_channels)
        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        batch_size, in_c, in_h, in_w = x.shape
        out_h = in_h - self.k + 1
        out_w = in_w - self.k + 1
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            patch = x[b, ic, i:i+self.k, j:j+self.k]
                            out[b, oc, i, j] += np.sum(patch * self.weights[oc, ic])
                out[b, oc] += self.bias[oc]
        return out

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, channels, h, w = x.shape
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        patch = x[b, c, start_i:start_i+self.size, start_j:start_j+self.size]
                        out[b, c, i, j] = np.max(patch)
        return out

class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias