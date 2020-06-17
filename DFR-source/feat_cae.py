import torch
import torch.nn as nn


class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50):
        super(FeatCAE, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels+2*latent_dim)//2, kernel_size=1, stride=1, padding=0)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels+2*latent_dim)//2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels+2*latent_dim)//2, kernel_size=1, stride=1, padding=0)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels+2*latent_dim)//2, in_channels, kernel_size=1, stride=1, padding=0)]
        # layers += [nn.ReLU()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss


if __name__ == "__main__":
    import numpy as np
    import time

    device = torch.device("cuda:1")
    x = torch.Tensor(np.random.randn(1, 3000, 64, 64)).to(device)
    feat_ae = FeatCAE(in_channels=3000, latent_dim=200).to(device)

    time_s = time.time()
    for i in range(10):
        time_ss = time.time()
        out = feat_ae(x)
        print("Time cost:", (time.time() - time_ss), "s")

    print("Time cost:", (time.time() - time_s), "s")
    print("Feature (n_samples, n_features):", out.shape)
