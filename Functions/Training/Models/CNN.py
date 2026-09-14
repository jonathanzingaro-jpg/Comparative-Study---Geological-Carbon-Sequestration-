import torch.nn as nn
import torch.nn.functional as F

# --------- Basic blocks ---------

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_c, out_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.scale = scale
        self.pad = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(in_c, out_c, 3, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = self.pad(x)
        x = self.relu(self.bn(self.conv(x)))
        return x


class CNN(nn.Module):
    """
    Input : (N, 12, 96, 200, 24)
    Output: (N,  1, 96, 200, 24)
    """

    def __init__(self, In_Channels, Out_Channels):
        super().__init__()

        # -------- Encoder --------
        self.enc1 = ConvBNReLU(In_Channels, 32, stride=2)   # -> (48,100,12)
        self.enc2 = ConvBNReLU(32, 64, stride=1)
        self.enc3 = ConvBNReLU(64, 128, stride=2)  # -> (24,50,6)
        self.enc4 = ConvBNReLU(128, 128, stride=1)
        self.enc5 = ConvBNReLU(128, 256, stride=2) # -> (12,25,3)
        self.enc6 = ConvBNReLU(256, 256, stride=1)

        # -------- Residual trunk --------
        self.resblocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(8)]
        )

        # -------- Decoder --------
        self.dec6 = ConvBNReLU(256, 256, stride=1)

        self.up5 = UpBlock(256, 256, scale=(2, 2, 2))  # -> (24,50,6)
        self.up4 = ConvBNReLU(256, 128, stride=1)

        self.up3 = UpBlock(128, 128, scale=(2, 2, 2))  # -> (48,100,12)
        self.up2 = ConvBNReLU(128, 64, stride=1)

        self.up1 = UpBlock(64, 32, scale=(2, 2, 2))    # -> (96,200,24)

        # -------- Output --------
        self.out = nn.Conv3d(32, Out_Channels, kernel_size=3, padding=1)

    def forward(self, x):

        # Modify Shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Encoder
        y = self.enc1(x)
        y = self.enc2(y)
        y = self.enc3(y)
        y = self.enc4(y)
        y = self.enc5(y)
        y = self.enc6(y)

        # Residual blocks
        y = self.resblocks(y)

        # Decoder
        y = self.dec6(y)
        y = self.up5(y)
        y = self.up4(y)
        y = self.up3(y)
        y = self.up2(y)
        y = self.up1(y)

        # Output
        y = self.out(y)

        # Modify Shape
        y = y.permute(0, 2, 3, 4, 1).contiguous().squeeze(-1)

        return y
