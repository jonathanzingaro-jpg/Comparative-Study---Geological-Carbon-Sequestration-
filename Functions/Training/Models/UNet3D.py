# Functions/Training/Models/UNet3D.py

import torch 
import torch.nn as nn
import torch.nn.functional as F

def DoubleConv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(inplace=True),
    )
    return conv

class Up3d(nn.Module):
    """Upscale by 2, concat skip, then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=(2,2,2), stride=(2,2,2))
        self.conv = DoubleConv(in_ch, out_ch)

    @staticmethod
    def _pad_to_match(x, ref):
        diffD = ref.size(2) - x.size(2)
        diffH = ref.size(3) - x.size(3)
        diffW = ref.size(4) - x.size(4)
        return F.pad(x, (diffW // 2, diffW - diffW // 2,
                         diffH // 2, diffH - diffH // 2,
                         diffD // 2, diffD - diffD // 2))

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResidualConv(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm3d(n_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm3d(n_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class UNet3D(nn.Module):
    def __init__(self, In_Channels, Out_Channels, Base_Channels, Bottleneck_Channels=None):
        super().__init__()

        BC = Base_Channels
        BN = Bottleneck_Channels if Bottleneck_Channels is not None else BC * 16

        # Shared Max Pooling
        self.Max_Pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # Encoder
        self.Conv_Initial = DoubleConv(In_Channels, BC)
        self.Down_Conv_1  = DoubleConv(BC,          BC * 2)
        self.Down_Conv_2  = DoubleConv(BC * 2,      BC * 4)
        self.Down_Conv_3  = DoubleConv(BC * 4,      BC * 8)

        # Bottleneck + Residual Connections
        self.bot  = DoubleConv(BC * 8, BN)
        self.res1 = ResidualConv(BN)
        self.res2 = ResidualConv(BN)
        self.res3 = ResidualConv(BN)
        self.res4 = ResidualConv(BN)
        self.res5 = ResidualConv(BN)
        self.res6 = ResidualConv(BN)

        # Decoder
        self.up3 = Up3d(BN,      BC * 8)
        self.up2 = Up3d(BC * 8,  BC * 4)
        self.up1 = Up3d(BC * 4,  BC * 2)
        self.up0 = Up3d(BC * 2,  BC)

        # Output
        self.Out = nn.Conv3d(BC, Out_Channels, kernel_size=1)

    def forward(self, x):

        # (Nb, Nx, Ny, Nt, Nc) → (Nb, Nc, Nx, Ny, Nt)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Encoder
        x0 = self.Conv_Initial(x)
        x1 = self.Down_Conv_1(self.Max_Pool(x0))
        x2 = self.Down_Conv_2(self.Max_Pool(x1))
        x3 = self.Down_Conv_3(self.Max_Pool(x2))

        # Bottleneck
        xb = self.bot(x3)
        xb = self.res1(xb)
        xb = self.res2(xb)
        xb = self.res3(xb)
        xb = self.res4(xb)
        xb = self.res5(xb)
        xb = self.res6(xb)

        # Decoder with skip connections
        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        # Output
        y = self.Out(y0)

        # (Nb, Nc, Nx, Ny, Nt) → (Nb, Nx, Ny, Nt) 
        y = y.permute(0, 2, 3, 4, 1).contiguous().squeeze(-1)

        return y