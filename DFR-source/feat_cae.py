import torch
import torch.nn as nn

#########################################
#    1 x 1 conv CAE
#########################################
class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
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


# #########################################
# #    3 x 3 conv CAE
# #########################################
# class FeatCAE(nn.Module):
#     """Autoencoder."""

#     def __init__(self, in_channels=1000, latent_dim=50):
#         super(FeatCAE, self).__init__()

#         layers = []
#         layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)]

#         self.encoder = nn.Sequential(*layers)

#         # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
#         # of the features for rgb
#         layers = []
#         layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=3, stride=1, padding=1)]
#         # layers += [nn.ReLU()]

#         self.decoder = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#     def relative_euclidean_distance(self, a, b):
#         return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

#     def loss_function(self, x, x_hat):
#         loss = torch.mean((x - x_hat) ** 2)
#         return loss

#     def compute_energy(self, x, x_hat):
#         loss = torch.mean((x - x_hat) ** 2, dim=1)
#         return loss

################################################
# Feature AE with Shuffle Group Convolution
################################################

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ChannelShuffle(nn.Module):
    def __init__(groups=1):
        self.groups = groups

    def forward(x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, groups, 
            channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

##########################################
# 3 x 3 group conv with shuffle CAE 
##########################################
class FeatSCAE(nn.Module):
    """Autoencoder with shuffled group convolution."""

    def __init__(self, in_channels=1000, latent_dim=50):
        """
        Note: in_channels and latent_dim has to be even, because we use shuffled group convolution
        """
        super(FeatCAE, self).__init__()
        
        self.groups = [8, 4]
        in_channels2 = (in_channels + 2 * latent_dim) // 2
        in_channels2 = in_channels2 + in_channels2%4
        # Encoder
        # inchannels should be a multiple of the number of groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels2, kernel_size=1, stride=1, padding=0, groups=8),
            nn.BatchNorm2d(num_features=in_channels2),
            nn.ReLU(inplace=True))
        self.channel_shuffle1 = ChannelShuffle(groups=8)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels2, 2 * latent_dim, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(inplace=True)
        )
        self.channel_shuffle2 = ChannelShuffle(groups=4)

        self.mid_conv = nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.conv3 = nn.Sequential(
            nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2 * latent_dim should be a multiple of the number of groups 4, if latent_dim is a multiple of 2 then it satisfies that condition
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2),
            nn.ReLU(inplace=True)
        )
        self.channel_shuffle4 = ChannelShuffle(groups=4)
        
        # (in_channels + 2 * latent_dim) // 2 should be a multiple of the number of groups 8
        self.conv5 = nn.Conv2d(inchannels4, in_channels, kernel_size=1, stride=1, padding=0, groups=8)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.channel_shuffle1(x)
        x = self.conv2(x)
        x = self.channel_shuffle2(x)
        x = self.mid_conv(x)

        # decoder
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.channel_shuffle4(x)
        x = self.conv5(x)
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
