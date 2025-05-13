from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense

class CNN:
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
