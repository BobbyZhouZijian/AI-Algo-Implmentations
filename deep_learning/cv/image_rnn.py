'''
Use RNN to do image classification.

Basic idea: Treat each row/column of the image pixels as a sequence
of inputs. Feed the inputs into a RNN and flatten the final weights
into the shape of the number of classes we want to classify. Then
backprop on the weight values.

Proved pretty efficient on some simple iamge classification tasks e.g. MNIST.
Accuracy for the code below: 98.6% with LSTM and 96.8% with RNN
'''

import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ImageRNN(nn.Module):
    def __init__(self, input_size, feature_size, num_classes, mode='LSTM'):
        super(ImageRNN, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        if mode == 'LSTM':
            self.rnn = nn.LSTM(input_size, feature_size)
        elif mode == 'RNN':
            self.rnn = nn.RNN(input_size, feature_size)
        else:
            raise ValueError('Mode argument not recognized')
        self.dropout = nn.Dropout(0.8)
        self.FC = nn.Linear(feature_size, num_classes)
    
    def forward(self, x):
        x = x.permute(1,0,2)

        if self.mode == 'LSTM':
            _, (hidden_state, cell_state) = self.rnn(x)
        elif self.mode == 'RNN':
            _, hidden_state = self.rnn(x)
        else:
            raise ValueError('Mode argument not recognized')
        out = self.dropout(hidden_state)
        out = self.FC(out)

        return out.view(-1, self.num_classes)


class ModelTrainer:
    def __init__(self, device='cuda'):
        self.model = None
        self.device = device
    
    def train(
        self,
        dataset,
        num_classes,
        num_epochs=25,
        feature_size = 150
    ):
        '''
        Train the RNN with the given torch dataset

        Parameter:
            a torch Dataset object
        '''


        # build model
        # input size fixed to 28
        input_size = 28
        self.model = ImageRNN(input_size, feature_size, num_classes).to(device=self.device)

        # build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=64,
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
                images = images.view(-1, 28, 28).to(self.device)
                labels = labels.to(self.device)

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
                images = images.view(-1,28,28).to(self.device)
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
    if os.path.exists(args.file_path+'/MNIST/'):
        download = False
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.MNIST(
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
        test_data = torchvision.datasets.MNIST(
            root=args.file_path,
            train=False,
            transform=transform,
            download=download
        )

        acc_score = net.test(test_data)
        print(f"accuracy score: {acc_score}")
    else:
        pass
            

