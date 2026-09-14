# Functions/Training/Models/3DCNN.py

# Import Libraries
from Functions.Import import *

# Import Libraries
from Functions.Import import *
import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Bigger 3D CNN for regression.
    Keeps spatial size the same with padding=1.
    Input:  (B, C, D, H, W)
    Output: (B, out_channels, D, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_channels, 64),


            block(64, 32),

            block(32, 16),

            nn.Conv3d(16, out_channels, kernel_size=1),
            nn.Identity(),  # regression output (no sigmoid)
        )

    def forward(self, x):
        return self.net(x)