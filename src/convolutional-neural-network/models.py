from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
import numpy as np

class AveragePool2D:
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
                        out[b, c, i, j] = np.mean(patch)
        return out

# Base CNN with 2 conv layers (original)
class CNN2Layer:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=16*6*6, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

# CNN with 3 conv layers
class CNN3Layer:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        
        self.conv3 = Conv2D(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=32*2*2, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

# CNN with 4 conv layers
class CNN4Layer:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        
        self.conv3 = Conv2D(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = ReLU()
        
        self.conv4 = Conv2D(in_channels=32, out_channels=64, kernel_size=3)
        self.relu4 = ReLU()
        self.pool4 = MaxPool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=64*2*2, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        
        x = self.conv4.forward(x)
        x = self.relu4.forward(x)
        x = self.pool4.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

# CNN with different filter counts
class CNNWideFilters:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=64*6*6, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

# CNN with different kernel sizes
class CNNLargeKernel:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=5)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=5)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=16*4*4, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x

# CNN with average pooling
class CNNAvgPool:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = AveragePool2D()
        
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)
        self.relu2 = ReLU()
        self.pool2 = AveragePool2D()
        
        self.flatten = Flatten()
        self.fc = Dense(input_dim=16*6*6, output_dim=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x 