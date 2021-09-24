"""
An encoder-decoder structure tries to encode a feature map to a lower dimensional representation
using the encoder network and then reconstructs the feature map with the decoder network.

Mathematically, the encoder: z = f(h_e(x))
                the decoder: x_hat = f(h_d(z))

The encoder and the decoder can be implemented with a neural network which learns the important features
to be encoded and decoded (auto encoder).

Reference: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self, in_features):
        super(AutoEncoder, self).__init__()

        # encoder layers
        self.in_1 = nn.Linear(in_features=in_features, out_features=256)
        self.in_2 = nn.Linear(in_features=256, out_features=128)

        # decoder layers
        self.out_1 = nn.Linear(in_features=128, out_features=256)
        self.out_2 = nn.Linear(in_features=256, out_features=in_features)

    def forward(self, x):
        x = self.in_1(x)
        x = F.relu(x)
        x = self.in_2(x)
        x = F.relu(x)

        x = self.out_1(x)
        x = F.relu(x)
        x = self.out_2(x)
        x = F.relu(x)

        return x


def train_model(num_epochs):
    # prepare data
    input_size = 784
    tf = transforms.Compose([transforms.ToTensor()])

    train_data = torchvision.datasets.MNIST(
        root='../data',
        train=True,
        transform=tf,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # train auto encoder
    for epoch in range(num_epochs):
        cum_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = data.view(-1, input_size).to(device)

            optimizer.zero_grad()
            output = model(data)

            # learn to reconstruct
            loss = criterion(output, data)
            loss.backward()

            optimizer.step()
            cum_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}; Iteration: {i + 1}; Avg loss: {cum_loss / (i+1)}.")

    return model


def evaluate(model):
    input_size = 784
    tf = transforms.Compose([transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = torchvision.datasets.MNIST(
        root='../data',
        train=False,
        transform=tf,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    outputs = []

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.view(-1, input_size).to(device)
            output = model(data).cpu().numpy()
            outputs.append(output)

    outputs = np.concatenate(outputs, axis=0)
    return outputs


if __name__ == '__main__':
    model = train_model(20)
    res = evaluate(model)

    # visualize a few images
    sample_img = res[0].reshape(28,28)
    plt.imshow(sample_img, interpolation='nearest')
    plt.show()
