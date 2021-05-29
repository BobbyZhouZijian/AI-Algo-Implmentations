'''
GoogLeNet was invented by Google used for image
classification.

It is well-known for its usage of Inception, a block
of convolutional layers which enables the network to be 
much deeper than vanilla CNN, hence achieving higher performance.

The code below is referenced from https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py
'''

import os
import argparse
from collections import namedtuple
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


__all__ = ['GoogleNet']


class BasicConv2d(nn.Module):
    '''a basic convolutional block'''

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        return F.relu(out, inplace=True)


class Inception(nn.Module):
    '''Build the Inception Block'''

    def __init__(
        self,
        in_channels,
        ch1_1,
        ch3_3red,
        ch3_3,
        ch5_5red,
        ch5_5,
        pool_proj,
        conv_block=None
    ):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3_3red, kernel_size=1),
            conv_block(ch3_3red, ch3_3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5_5red, kernel_size=1),
            conv_block(ch5_5red, ch5_5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
    
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (4,4))
        # size: in_channels * 4 * 4
        out = self.conv(out)
        # size: 128 * 4 * 4
        out = torch.flatten(out, 1)
        # size: 2048
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, 0.7, training=self.training)

        out = self.fc2(out)
        return out

    

class GoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        aux_logits=False,
        transform_input=False
    ):
        super(GoogLeNet, self).__init__()
        blocks = [BasicConv2d, Inception, InceptionAux]

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)


        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None
        
        self.avgppol = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
    
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:,0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:,1], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:,2], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
    
    def _forward(self, x):
        # 3 * 224 * 224
        x = self.conv1(x)
        # 64 * 112 * 112
        x = self.maxpool1(x)
        # 64 * 56 * 56
        x = self.conv2(x)
        # 64 * 56 * 56
        x = self.conv3(x)
        # 192 * 56 * 56
        x = self.maxpool2(x)
        # 192 * 28 * 28

        x = self.inception3a(x)
        # 256 * 28 * 28
        x = self.inception3b(x)
        # 480 * 28 * 28
        x = self.maxpool3(x)
        # 480 * 14 * 14
        x = self.inception4a(x)
        # 512 * 14 * 14
        aux1 = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        # 512 * 14 * 14
        x = self.inception4c(x)
        # 512 * 14 * 14
        x = self.inception4d(x)
        # 528 * 14 * 14

        aux2 = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # 832 * 14 * 14
        x = self.maxpool4(x)
        # 832 * 7 * 7
        x = self.inception5a(x)
        # 832 * 7 * 7
        x = self.inception5b(x)
        # 1024 * 7 * 7

        x = self.avgppol(x)
        # 1024 * 1 * 1
        x = torch.flatten(x, 1)
        # 1024
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux2, aux1
    

    def forward(self, x):
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)

        aux_defined = self.training and self.aux_logits
        if aux_defined:
            GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x


class ModelTrainer:
    def __init__(self, aux_logits=True, transform_inputs=True, device='cuda'):
        self.aux_logits = aux_logits,
        self.transform_inputs = transform_inputs
        self.model = None
        self.device = device
    
    def train(
        self,
        dataset,
        num_classes,
        num_epochs=15,
    ):
        '''
        Train the CNN with the given torch dataset

        Parameter:
            a torch Dataset object
        '''


        # build model
        self.model = GoogLeNet(
            num_classes,
            in_channels=3,
            aux_logits=self.aux_logits,
            transform_input=self.transform_inputs
        ).to(device=self.device)

        # build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True
        )

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        num_steps = len(data_loader)

        self.model.train()

        for t in range(num_epochs):
            start = time.time()
            for i, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.aux_logits:
                    outputs, aux1, aux2 = self.model(images)
                    loss = criterion(outputs, labels) \
                        + 0.3 * criterion(aux1, labels) \
                            + 0.3 * criterion(aux2, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if (i+1) % 100 == 0:
                    print(f"Epoch: {t+1}/{num_epochs}, Step: {i+1}/{num_steps}, Loss: {loss.item()}")
            end = time.time()
            print(f'Time elapsed for Epoch [{t+1}/{num_epochs}]: {end - start}s')
    
    def infer(self, images):
        '''
        infer the given image set

        Parameter:
            a torch Dataset object for inference
        
        output:
            a list consisting of the predicted classes
        '''
        images = images.to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.aux_logits:
                outputs, _, _ = self.model(images)
            else:
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
    if os.path.exists(args.file_path+'/cifar-10-batches-py/'):
        download = False
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
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
            

