"""
Variational Auto-Encoder has a similar structure with a traditional autoencoder,
but has very different mathematical underpinning.

Ideally, we want to sample latent variables z from a distribution P(z|X) so that
z can very likely produce an output similar to X:

    D(P(z|X) || Q(z)) = E[log Q(z) - log P(X|z) + log P(z)] + log P(x)

This can be rearranged asL:

    log P(x) - D(P(z|X) || Q(z)) = E[log P(X|z)] - D(Q(z|X) || P(z))

The lhs is what we want to maximize. So we can do SGD to the rhs.

Reference: https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
           https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # encoder and decoder
        self.enc_1 = nn.Linear(input_dim, 256)

        self.dec_1 = nn.Linear(latent_dim, 256)
        self.dec_2 = nn.Linear(256, input_dim)

        # mu and var
        self.mu = nn.Linear(256, latent_dim)
        self.var = nn.Linear(256, latent_dim)

    def encode(self, x):
        x = self.enc_1(x)
        x = F.relu(x)
        return self.mu(x), self.var(x)

    def decode(self, x):
        x = self.dec_1(x)
        x = F.relu(x)
        x = self.dec_2(x)
        x = F.relu(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples, device=torch.device('cpu')):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)

        return x_hat, x, mu, log_var


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
    model = VAE(784)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        cum_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            x_hat, x, mu, log_var = model(data)

            optimizer.zero_grad()

            # not quite what E[log P(X|z)] is about but works well in practice
            recons_loss = F.mse_loss(x_hat, x)

            # KL Divergence loss
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

            loss = (kl_loss + recons_loss).mean()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}; Iteration: {i + 1}; Avg loss: {cum_loss / (i + 1)}.")

    print("finished training!")
    return model


def test(model, num_samples=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        samples = model.sample(num_samples, device)
        sample = samples[0].cpu().numpy().reshape(28, 28)

    plt.imshow(sample, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    model = train_model(5)
    test(model)
