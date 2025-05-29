import numpy as np

class Conv2DLayer:
    def __init__(self, weights, biases, activation='relu', padding='same'):
        self.weights = weights  # Shape: (kernel_h, kernel_w, input_channels, output_channels)
        self.biases = biases    # Shape: (output_channels,)
        self.activation = activation
        self.padding = padding
    
    def forward(self, x):
        # Implementasi konvolusi 2D
        batch_size, input_h, input_w, input_channels = x.shape
        kernel_h, kernel_w, _, output_channels = self.weights.shape
        
        # Padding
        if self.padding == 'same':
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            x_padded = x
            
        # Output dimensions
        output_h = input_h if self.padding == 'same' else input_h - kernel_h + 1
        output_w = input_w if self.padding == 'same' else input_w - kernel_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, output_h, output_w, output_channels))
        
        # Perform convolution
        for b in range(batch_size):
            for h in range(output_h):
                for w in range(output_w):
                    for c in range(output_channels):
                        # Extract patch
                        patch = x_padded[b, h:h+kernel_h, w:w+kernel_w, :]
                        # Compute convolution
                        output[b, h, w, c] = np.sum(patch * self.weights[:, :, :, c]) + self.biases[c]
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
            
        return output

class MaxPooling2DLayer:
    def __init__(self, pool_size=2, strides=2):
        self.pool_size = pool_size
        self.strides = strides
    
    def forward(self, x):
        batch_size, input_h, input_w, channels = x.shape
        output_h = input_h // self.strides
        output_w = input_w // self.strides
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for h in range(output_h):
                for w in range(output_w):
                    for c in range(channels):
                        h_start = h * self.strides
                        h_end = h_start + self.pool_size
                        w_start = w * self.strides
                        w_end = w_start + self.pool_size
                        
                        patch = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.max(patch)
        
        return output

class AveragePooling2DLayer:
    def __init__(self, pool_size=2, strides=2):
        self.pool_size = pool_size
        self.strides = strides
    
    def forward(self, x):
        batch_size, input_h, input_w, channels = x.shape
        output_h = input_h // self.strides
        output_w = input_w // self.strides
        
        output = np.zeros((batch_size, output_h, output_w, channels))
        
        for b in range(batch_size):
            for h in range(output_h):
                for w in range(output_w):
                    for c in range(channels):
                        h_start = h * self.strides
                        h_end = h_start + self.pool_size
                        w_start = w * self.strides
                        w_end = w_start + self.pool_size
                        
                        patch = x[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.mean(patch)
        
        return output

class DenseLayer:
    def __init__(self, weights, biases, activation=None):
        self.weights = weights  # Shape: (input_features, output_features)
        self.biases = biases    # Shape: (output_features,)
        self.activation = activation
    
    def forward(self, x):
        # Linear transformation
        output = np.dot(x, self.weights) + self.biases
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            # Apply softmax
            exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
        return output

class FlattenLayer:
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)