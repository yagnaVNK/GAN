import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,latent_dim ,total_filters,image_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.total_filters = total_filters
        self.image_channels = image_channels

        self.layers= nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.total_filters * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.total_filters * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.total_filters * 8, self.total_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.total_filters * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.total_filters * 4, self.total_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.total_filters * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.total_filters * 2, self.total_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.total_filters),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.total_filters, self.image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, input):
        output = self.layers(input)
        return output