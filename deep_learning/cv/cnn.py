'''
Convolutional Neural Network is a network designed for image
classification. In this task, we build a simple CNN to learn
on MNIST dataset.

A classic CNN is typically comprised of several layers. Each layer is
formed by a stack of a convolutional layer, a batch normalizaation,
an acivation and pooling lyaer.

After the convolutional layers, in order to get a classification score,
we pass the last conv layer to a few fully connected layers.
'''

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from .activations.sigmoid import Sigmoid

__all__ = ['CNN']

class ConvNet(nn.Module):
    '''Convolutional Network Model builder'''
    def __init__(self, H, W, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            # use hard sigmoid to speed up training
            Sigmoid(hard=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # calcualte new H and new W
        H = H // 2
        W = W // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            Sigmoid(hard=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        H = H // 2
        W = W // 2

        self.fc = nn.Linear(H * W * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

class CNN:
    def __init__(self, input_h, input_w, device='cuda'):
        self.model = None
        self.H = input_h
        self.W = input_w
        self.device = device
    
    def train(self, dataset, lr=0.01, num_epochs=25):
        '''
        Train the CNN with the given torch dataset

        Parameter:
            a torch Dataset object
        '''

        # build model
        self.model = ConvNet(self.H, self.W).to(self.device)

        # build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True
        )

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        num_steps = len(data_loader)

        for t in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # for debug
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if (i+1) % 100 == 0:
                    print(f"epoch: {t+1}/{num_epochs}, step: {i+1}/{num_steps}, loss: {loss.item()}")
    
    def infer(self, images):
        '''
        infer the given image set

        Parameter:
            a torch Dataset object for inference
        
        output:
            a list consisting of the predicted classes
        '''
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            return outputs.cpu().data
    
    def test(self, data):
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=32,
            shuffle=False
        )

        self.model.eval()
        num_hit = 0
        total_num = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                # get the max likelihood prediction for each sample
                _, predicted = torch.max(outputs.data, 1)
                total_num += len(labels)
                num_hit += (predicted == labels).sum().item()
        
        return num_hit / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='training data file path')
    parser.add_argument('--eval_mode', action='store_true', help='run this in evaluation mode')
    args = parser.parse_args()

    download = True
    if os.path.exists(args.file_path+'/CIFAR10/'):
        download = False

    train_data = torchvision.datasets.CIFAR10(
        root=args.file_path,
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} as the training device...')
    net = CNN(32, 32, device=device)
    net.train(train_data)
    if args.eval_mode:
        test_data = torchvision.datasets.CIFAR10(
            root=args.file_path,
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )

        acc_score = net.test(test_data)
        print(f"accuracy score: {acc_score}")
    else:
        pass
            
