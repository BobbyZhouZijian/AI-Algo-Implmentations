"""
PixelCNN is a belief update network used to generate images similar to the training images.
Implementation details focus on the idea of a masked ConvNet filter. The PixelCNN network
outputs a conv layer with 256 channels, corresponding to the 256 pixel values.

Reference: https://github.com/singh-hrituraj/PixelCNN-Pytorch
"""

import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision

sys.path.append('../')
from optimizers.lars import LARS


class MaskedConvNet(nn.Conv2d):
    def __init__(self, is_first_layer, *args, **kwargs):
        super(MaskedConvNet, self).__init__(*args, **kwargs)
        # use register_buffer so that the mask will not be updated by the optimizer
        self.register_buffer('mask', self.weight.data.clone())
        b, c, h, w = self.weight.size()
        self.mask.fill_(1)
        if is_first_layer:
            self.mask[:, :, h // 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConvNet, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, no_layers=8, kernel_size=7, channels=64):
        super(PixelCNN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(no_layers):
            if i == 0:
                self.layers.append(MaskedConvNet(True, 1, channels, kernel_size, 1, kernel_size // 2, bias=False))
            else:
                self.layers.append(
                    MaskedConvNet(False, channels, channels, kernel_size, 1, kernel_size // 2, bias=False))
            self.layers.append(nn.BatchNorm2d(channels))
            self.layers.append(nn.ReLU(inplace=True))

        # last conv output 256 channels, representing prob for each pixel
        self.layers.append(nn.Conv2d(channels, 256, 1))

    def forward(self, x):
        # forward x layers
        for layer in self.layers:
            x = layer(x)
        return x


def train(dataloader, lr=1e-3, num_epochs=5, milestones=None, device='cpu', save_path=None):
    model = PixelCNN().to(device)

    # try out my own implementation of LARS optimizer, combined with a scheduler
    optimizer = LARS(model.parameters(), lr=lr)
    if milestones is None:
        milestones = [num_epochs]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    num_steps = len(dataloader)

    model.train()
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(dataloader):
            images = images.float().to(device)
            target = Variable(images[:, 0, :, :] * 255).long()

            optimizer.zero_grad()
            out = model(images)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}; Step: {step + 1}/{num_steps}; Loss: {loss.item()}")

        scheduler.step()

    print("Training finished!\n")

    if save_path is not None:
        file_path = save_path + '/epoch_' + str(num_epochs) + '.pth'
        print(f"saving model to {file_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), file_path)

    return model


def generate(model, save_path='./saved_img.png', channels=1, im_size=28, n_row=12, device='cpu'):
    model.eval()

    # plain input to be updated

    sample = torch.zeros(n_row * n_row, channels, im_size, im_size).to(device)

    print("Starting to generate image.\n")
    with torch.no_grad():
        for i in tqdm(range(im_size)):
            for j in range(im_size):
                out = model(sample)
                probs = F.softmax(out[:, :, i, j], -1).data
                # sample from the probability distribution, then normalize it
                # alternative: just take the max
                # sample[:,:,i,j] = torch.argmax(probs).float() / 255.0
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0

    # save image
    print(f"Generated image! Saved at {save_path}.\n")
    torchvision.utils.save_image(sample, save_path, nrow=n_row, padding=0)


if __name__ == '__main__':
    # test with MNIST
    train_data = torchvision.datasets.MNIST(
        root='../../data',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # use large batch size since images are pretty small and I wanna try out LARS haha
    batch_size = 96
    load_path = None  # './pixelcnn_models/epoch_6.pth'

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    if load_path is not None:
        print(f"loading existing model")
        model = PixelCNN().to(device)
        model.load_state_dict(torch.load(load_path))
    else:
        print(f"using {device} as training device...")
        model = train(train_loader, lr=0.05, device=device, num_epochs=25, milestones=[10, 20],
                      save_path='./pixelcnn_models')
    generate(model, device=device)
