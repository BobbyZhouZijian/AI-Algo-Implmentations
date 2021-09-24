import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


class GeneratorNet(nn.Module):
    def __init__(self, img_size, num_channels, latent_size):
        super(GeneratorNet, self).__init__()
        self.init_size = img_size // 4
        self.linear = nn.Linear(latent_size, 128 * (self.init_size ** 2))
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        return self.conv(x)


def conv_block(in_size, out_size, need_bn):
    conv = nn.Sequential(
        nn.Conv2d(in_size, out_size, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(0.2),
    )
    if need_bn:
        conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_size)
        )
    return conv


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DiscriminatorNet(nn.Module):
    def __init__(self, num_channels, img_size):
        super(DiscriminatorNet, self).__init__()
        self.num_channels = num_channels
        self.img_size = img_size

        self.layers = nn.Sequential(
            conv_block(num_channels, 16, False),
            conv_block(16, 32, True),
            conv_block(32, 64, True),
            conv_block(64, 128, True),
        )

        self.linear = nn.Linear(128 * (img_size // 16)**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return self.sigmoid(x)


def train(data_loader, img_size, in_channels, lr=2e-4, num_epochs=10, device='cpu'):
    latent_size = 100
    generator = GeneratorNet(img_size, in_channels, latent_size).to(device)
    discriminator = DiscriminatorNet(in_channels, img_size).to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    g_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    num_steps = len(data_loader)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.float().to(device)
            real_target = torch.Tensor(images.shape[0], 1).fill_(1.0).to(device)
            fake_target = torch.Tensor(images.shape[0], 1).fill_(0.0).to(device)

            g_opt.zero_grad()
            z = torch.normal(0, 1, size=(images.shape[0], latent_size)).to(device)
            gen = generator(z)

            g_loss = criterion(discriminator(gen), real_target)
            g_loss.backward()
            g_opt.step()

            d_opt.zero_grad()
            real_loss = criterion(discriminator(images), real_target)
            fake_loss = criterion(discriminator(gen.detach()), fake_target)
            d_loss = (real_loss + fake_loss) / 2.0
            d_loss.backward()
            d_opt.step()

            if (i+1) % 200 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, step: {i+1}/{num_steps},"
                      + f" g_loss: {g_loss.item()}, d_loss: {d_loss.item()}")

    # save models
    print("saving models")
    gen_state = generator.state_dict()
    disc_state = discriminator.state_dict()
    torch.save(gen_state, './gan_models/gen_model_2.pth')
    torch.save(disc_state, './gan_models/disc_model_2.pth')
    print("models saved")


def test(model, latent_size):
    with torch.no_grad():
        z = torch.normal(0, 1, size=(16, latent_size)).to(device)
        gen = model(z)
        torchvision.utils.save_image(gen, './gen.png', nrow=4, padding=0, normalize=True)


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(
        root='../../data/',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using {device} as training device")
    train(train_loader, 32, 3, num_epochs=100, device=device)
    latent_size = 100
    model = GeneratorNet(32, 3, latent_size).to(device)
    state = torch.load('./gan_models/gen_model_2.pth')
    model.load_state_dict(state)
    test(model, latent_size)