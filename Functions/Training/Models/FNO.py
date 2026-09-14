# Functions/Training/Models/FNO.py

# Import Libraries
import torch

# Fourier Spectral Layer
class SpectralConv3D(torch.nn.Module):

    def __init__(self, In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3):
        super().__init__()

        self.In_Channels = In_Channels                                                                                             # Number of Input Channels
        self.Out_Channels = Out_Channels                                                                                           # Number of Output Channels
        self.Modes_1 = Modes_1                                                                                                     # Number of Fourier Modes in x
        self.Modes_2 = Modes_2                                                                                                     # Number of Fourier Modes in y
        self.Modes_3 = Modes_3                                                                                                     # Number of Fourier Modes in t
        self.scale = 1 / (In_Channels * Out_Channels)                                                                              # Scaling Factor for Weights

        # Weight Matrices in Fourier Space -> (In_Channels, Out_Channels, N_Modes)
        self.W1 = torch.nn.Parameter(self.scale * torch.rand(In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3, dtype=torch.cfloat))  
        self.W2 = torch.nn.Parameter(self.scale * torch.rand(In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3, dtype=torch.cfloat))  
        self.W3 = torch.nn.Parameter(self.scale * torch.rand(In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3, dtype=torch.cfloat))  
        self.W4 = torch.nn.Parameter(self.scale * torch.rand(In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3, dtype=torch.cfloat))  

    # Complex Multiplication
    def Complex_Multi_3d(self, input, weights):
        # (N_Batch, In_Channels, Nx, Ny, Nt), (In_Channels, Out_Channels, Modes_x, Modes_y, Modes_t) -> (N_Batch, Out_Channels, Modes_x, Modes_y, Modes_t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):

        # Compute Fourier Coefficients 
        x_FT = torch.fft.rfftn(x, dim = [-3,-2,-1])

        # Define Empty Tensor for Fourier Coefficient Products with Weights
        Out_FT = torch.zeros(x.shape[0], self.Out_Channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # Multiplication of Weights and Fourier Coefficients 
        Out_FT[:, :, :self.Modes_1, :self.Modes_2, :self.Modes_3] = self.Complex_Multi_3d(x_FT[:, :, :self.Modes_1, :self.Modes_2, :self.Modes_3], self.W1)      
        Out_FT[:, :, -self.Modes_1:, :self.Modes_2, :self.Modes_3] = self.Complex_Multi_3d(x_FT[:, :, -self.Modes_1:, :self.Modes_2, :self.Modes_3], self.W2)    
        Out_FT[:, :, :self.Modes_1, -self.Modes_2:, :self.Modes_3] = self.Complex_Multi_3d(x_FT[:, :, :self.Modes_1, -self.Modes_2:, :self.Modes_3], self.W3)    
        Out_FT[:, :, -self.Modes_1:, -self.Modes_2:, :self.Modes_3] = self.Complex_Multi_3d(x_FT[:, :, -self.Modes_1:, -self.Modes_2:, :self.Modes_3], self.W4)  

        # Inverse Fourier Transform 
        y = torch.fft.irfftn(Out_FT, s=(x.size(-3), x.size(-2), x.size(-1)))

        return y

class FNO(torch.nn.Module):

    def __init__(self, In_Channels, Out_Channels, Modes_1, Modes_2, Modes_3, N_W, N_F):
        super().__init__()

        # Define Number of Fourier Modes in X, Y, t
        self.Modes1 = Modes_1
        self.Modes2 = Modes_2
        self.Modes3 = Modes_3

        # Define Input and Output Shapes
        self.In_Channels = In_Channels
        self.Out_Channels = Out_Channels

        # Number of Neurons in Fourier Space
        self.N_F = N_F
        self.N_W = N_W

        # Define Feedforward Layers
        self.FC0 = torch.nn.Linear(In_Channels, N_W)      # Lifting Layer
        self.FC1 = torch.nn.Linear(N_W, N_F)              # Feedforward Layer before Output
        self.FC2 = torch.nn.Linear(N_F, Out_Channels)  # Projection for Output

        # Define Spectral Convolution Layers
        self.conv1 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)
        self.conv2 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)
        self.conv3 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)
        self.conv4 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)
        self.conv5 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)
        self.conv6 = SpectralConv3D(self.N_W, self.N_W, self.Modes1, self.Modes2, self.Modes3)

        # Define 1D Convolution Layers
        self.w1 = torch.nn.Conv1d(self.N_W, self.N_W, 1)
        self.w2 = torch.nn.Conv1d(self.N_W, self.N_W, 1)
        self.w3 = torch.nn.Conv1d(self.N_W, self.N_W, 1)
        self.w4 = torch.nn.Conv1d(self.N_W, self.N_W, 1)
        self.w5 = torch.nn.Conv1d(self.N_W, self.N_W, 1)
        self.w6 = torch.nn.Conv1d(self.N_W, self.N_W, 1)

    def forward(self, x):

        # Tensor Sizes
        N_Batch = x.shape[0]
        Nx_0, Ny_0, Nt_0 = x.shape[1], x.shape[2], x.shape[3]

        # Pad Edges 
        x = torch.nn.functional.pad(torch.nn.functional.pad(x, (0,0,0,8,0,8), "replicate"), (0,0,0,0,0,0,0,8), 'constant', 0)
        N_x, N_y, N_t = x.shape[1], x.shape[2], x.shape[3]

        # Run through Lifting Layer [N_Batch, N_x, N_y, N_z, In_Channels] -> [N_Batch, N_W, N_x, N_y, N_z]
        x = self.FC0(x).permute(0, 4, 1, 2, 3).contiguous()

        # Fourier Layer 1 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv1(x)
        x2 = self.w1(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Fourier Layer 2 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv2(x)
        x2 = self.w2(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Fourier Layer 3 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv3(x)
        x2 = self.w3(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Fourier Layer 4 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv4(x)
        x2 = self.w4(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Fourier Layer 5 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv5(x)
        x2 = self.w5(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Fourier Layer 6 -> [N_Batch, N_W, N_x, N_y, N_z]
        x1 = self.conv6(x)
        x2 = self.w6(x.view(N_Batch, self.N_W, -1)).view(N_Batch, self.N_W, N_x, N_y, N_t)
        x = x1 + x2
        x = torch.nn.functional.relu(x)

        # Feedforward Layer with Projection 
        x = torch.nn.functional.relu(self.FC1(x.permute(0, 2, 3, 4, 1).contiguous())) # [N_Batch, N_x, N_y, N_z, N_F]   
        x = self.FC2(x)  # [N_Batch, N_x, N_y, N_z, Out_Channels]   

        # Remove Boundary Padding 
        x = x[:, :Nx_0, :Ny_0, :Nt_0, :]

        return x.squeeze(-1)