import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,latent_dim ,total_filters,image_channels):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            
            nn.Conv2d(image_channels, total_filters, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(total_filters, total_filters * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(total_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(total_filters * 2, total_filters * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(total_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(total_filters * 4, total_filters * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(total_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(total_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.layers(input)
        return output.view(-1, 1).squeeze(1)