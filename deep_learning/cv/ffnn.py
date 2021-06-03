'''
Feed Forward Neural Network is a basic form of a deep neural network.
It comprises of a few layers that pass down in a feed forward manner to
the output layer.

In this implementation we use a 3 layer structure with PyTorch
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import get_input_label_split, get_accuracy, get_precision
import argparse
import numpy as np
import pandas as pd

__all__ = ['FFNN']

class HeartTrainDataset(Dataset):
    def __init__(self, data, label_name):
        self.X, self.y = get_input_label_split(data, label_name)
    
    def __getitem__(self, index):
        features = self.X[index]
        feat_tensor = torch.tensor(features, dtype=torch.float32)

        label = self.y[index]
        label = np.expand_dims(label, axis=0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return feat_tensor, label_tensor
    
    def __len__(self):
        return len(self.X)

class HeartTestDataset(Dataset):
    def __init__(self, data):
        self.X = data
    
    def __getitem__(self, index):
        features = self.X[index]
        return torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)


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

        dataset = HeartTrainDataset(data, label_name)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=True,
        )

        input_size = data.shape[1] - 1
        self.model = NeuralNet(input_size, 128, 128, 1).to(self.device)

        # loss and optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        num_steps = len(train_loader)

        for t in range(num_epochs):
            for i, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                # backward prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 4 == 0:
                    print(f"Epoch {t}: {i}/{num_steps} steps have elasped. Current loss {loss.item()}")
    

    def infer(self, data):
        test_x = HeartTestDataset(data)
        test_loader = DataLoader(
            dataset=test_x,
            batch_size=16,
            shuffle=False
        )

        with torch.no_grad():
            pred = []
            for features in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                predicted = np.sign(outputs.cpu().data)
                pred += predicted
            return pred


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