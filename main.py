import torch
import torch.nn as nn
from Generator import Generator as G
from torchsummary import summary


latent_dim = 100
total_filters = 64
image_channels = 3

generator = G(latent_dim=latent_dim,total_filters=total_filters,image_channels=image_channels)

