"""
Implementation of ResNet based on the paper:
    https://arxiv.org/pdf/1512.03385.pdf

Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            downsample=None,
            norm_layer=None
    ):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        out = self.relu(x)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes=10,
            norm_layer=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = self.norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample, norm_layer=norm_layer)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ModelTrainer:
    def __init__(self, device='cuda'):
        self.model = None
        self.device = device

    def train(
            self,
            dataset,
            num_classes,
            num_epochs=10,
    ):
        '''
        Train the ResNet with the given torch dataset

        Parameter:
            a torch Dataset object
        '''

        # build model
        self.model = ResNet(
            ResidualBlock,
            [2, 2, 2],
            num_classes,
        ).to(device=self.device)

        # build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=96,
            shuffle=True
        )

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-3, weight_decay=1e-5)

        num_steps = len(data_loader)

        self.model.train()

        for t in range(num_epochs):
            start = time.time()
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch: {t + 1}/{num_epochs}, Step: {i + 1}/{num_steps}, Loss: {loss.item()}")
            end = time.time()
            print(f'Time elapsed for Epoch [{t + 1}/{num_epochs}]: {end - start}s')

    def infer(self, images):
        """
        infer the given image set

        Parameter:
            a torch Dataset object for inference

        output:
            a list consisting of the predicted classes
        """
        images = images.to(self.device)
        self.model.eval()
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
    if os.path.exists(args.file_path + '/cifar-10-batches-py/'):
        download = False

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=args.file_path,
        train=True,
        transform=transform,
        download=download
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} as the training device...')
    net = ModelTrainer(device=device)
    net.train(train_data, 10)
    if args.eval_mode:
        test_data = torchvision.datasets.CIFAR10(
            root=args.file_path,
            train=False,
            transform=transform,
            download=download
        )

        acc_score = net.test(test_data)
        print(f"accuracy score: {acc_score}")
    else:
        pass
