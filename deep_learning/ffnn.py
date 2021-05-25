'''
Feed Forward Neural Network is a basic form of a deep neural network.
It comprises of a few layers that pass down in a feed forward manner to
the output layer.

In this implementation we use a 3 layer structure with PyTorch
'''

import torch
import torch.nn as nn
from util import get_input_label_split, get_accuracy, get_precision
import argparse
import numpy as np
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
        

class FFNN:
    def __init__(self, device='cpu'):
        self.train_x = None
        self.train_y = None
        self.model = None
        self.device = device
    
    def train(self, data, label_name, lr=7e-4, num_epochs=5):
        train_x, train_y = get_input_label_split(data, label_name)

        # convert training data to torch
        self.train_x = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        self.train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)

        input_size = self.train_x.shape[1]
        self.model = NeuralNet(input_size, 128, 128, 1).to(self.device)

        # loss and optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        num_steps = len(self.train_x)

        for t in range(num_epochs):
            for i in range(num_steps):
                features = self.train_x[i].reshape(1,-1).to(self.device)
                label = self.train_y[i].reshape(1,-1).to(self.device)

                output = self.model(features)
                loss = criterion(output, label)

                # backward prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch {t}: {i} steps have elasped. Current loss {loss.item()}")
    

    def infer(self, data):
        test_x = torch.tensor(data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(test_x)
            predicted = np.sign(outputs.cpu().data)
            return predicted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='training data file path')
    parser.add_argument('--label_name', type=str, default='label', help='label column name for the input file')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    if args.eval_mode:
        train_sz = int(len(df) * 0.8)
        df_train = df[:train_sz]
        df_test = df[train_sz:]
        net = FFNN(device='cuda')
        net.train(df_train, args.label_name)
        test_x, test_y = get_input_label_split(df_test, args.label_name)
        pred = net.infer(test_x)

        print(f"accuracy score: {get_accuracy(pred, test_y)}")
        print(f"precision score: {get_precision(pred, test_y)}")

    else:
        pass