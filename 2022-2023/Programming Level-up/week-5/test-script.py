# internal imports
import math
import argparse

# external imports
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    return parser.parse_args()


def load_datasets(batch_size):
    transforms = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train = MNIST(root="data/", train=True, download=True, transform=transforms)
    test = MNIST(root="data/", train=False, download=True, transform=transforms)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, optimiser, train_loader, device=torch.device("cpu"), epoch_num=0):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        optimiser.zero_grad()
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = torch.nn.functional.nll_loss(pred, target)
        loss.backward()
        optimiser.step()
        if idx % 100 == 0:
            print(f"[Epoch {epoch_num} ({idx}/{math.ceil(len(train_loader.dataset)/train_loader.batch_size)})]: {loss.item()}")


def test(model, test_loader, device=torch.device("cpu")):
    model.eval()
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        pred = pred.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return (correct / len(test_loader.dataset))*100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, 1)
        return x


def main(config):
    train_loader, test_loader = load_datasets(config.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimiser = torch.optim.Adadelta(model.parameters(), config.learning_rate)
    scheduler = StepLR(optimiser, step_size=1, gamma=0.7)
    
    print(f"Running on {device}")

    for i in range(1, config.epochs+1):
        train_loss = train(model, optimiser, train_loader, device, i)
        test_acc = test(model, test_loader, device)
        print(f"End of epoch {i}. Testing accuracy: {test_acc}")


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
