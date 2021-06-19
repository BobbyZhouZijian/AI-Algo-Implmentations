"""
PixelCNN is a belief update network used to generate images similar to the training images.
Implementation details focus on the idea of a masked ConvNet filter. The PixelCNN network
outputs a conv layer with 256 channels, corresponding to the 256 pixel values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm


class MaskedConvNet(nn.Conv2d):
    def __init__(self, device, is_first_layer=False, *args, **kwargs):
        super(MaskedConvNet, self).__init__(*args, **kwargs)
        self.mask = self.weight.data.clone().to(device)
        b, c, h, w = self.weight.size()
        self.mask.fill_(1)
        if is_first_layer:
            self.mask[:,:,h//2,w//2:] = 0
        else:
            self.mask[:,:,h//2+1,w//2:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConvNet, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, no_layers=8, kernel_size=7, channels=64, device='cpu'):
        super(PixelCNN, self).__init__()
        self.no_layers = no_layers
        self.kernel_size = kernel_size
        self.channels = channels

        self.conv_1 = MaskedConvNet(device, True, 1, channels, kernel_size, 1, kernel_size//2)
        self.conv_2 = MaskedConvNet(device, False, channels, channels, kernel_size, 1, kernel_size//2)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # last conv output 256 channels, representing prob for each pixel
        self.conv_3 = nn.Conv2d(channels, 256, 1)

    def forward(self, x):
        # forward x layers
        for i in range(self.no_layers):
            if i == 0:
                x = self.conv_1(x)
            else:
                x = self.conv_2(x)
            x = self.bn(x)
            x = self.relu(x)
        return self.conv_3(x)


def train(dataloader, lr=0.01, num_epochs=5, device='cpu'):
    model = PixelCNN(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_steps = len(dataloader)

    model.train()
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(dataloader):
            images = images.float().to(device)
            target = Variable(images[:,0,:,:] * 255).long()

            optimizer.zero_grad()
            out = model(images)

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if (step+1) % 100 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}; Step: {step+1}/{num_steps}; Loss: {loss.item()}")

    print("Training finished!\n")
    return model


def generate(model, save_path='./saved_img.png', channels=1, im_size=28, n_row=3, device='cpu'):
    model.eval()

    # plain input to be updated

    sample = torch.zeros(n_row*n_row, channels, im_size, im_size).to(device)

    print("Starting to generate image.\n")
    with torch.no_grad():
        for i in tqdm(range(im_size)):
            for j in range(im_size):
                out = model(sample)
                probs = F.softmax(out[:,:,i,j], -1).data
                # sample from the probability distribution, then normalize it
                # alternative: just take the max
                # sample[:,:,i,j] = torch.argmax(probs).float() / 255.0
                sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

    # save image
    print(f"Generated image! Saved at {save_path}.\n")
    torchvision.utils.save_image(sample, save_path, n_rows=n_row, padding=0)


if __name__ == '__main__':
    # test with MNIST
    train_data = torchvision.datasets.MNIST(
        root='../../data',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    print(f"using {device} as training device...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = train(train_loader, device=device, num_epochs=20)
    generate(model, device=device)
