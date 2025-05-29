from utils import load_cifar10
from model import CNN
from experiment import main as run_experiments

x_train, y_train, x_val, y_val = load_cifar10()

model = CNN()

output = model.forward(x_train[:10]) 

print("Output shape:", output.shape)

if __name__ == "__main__":
    run_experiments()
