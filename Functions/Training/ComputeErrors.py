# Functions/Training/ComputeError.py

# Import Libraries
from Functions.Import import *

# Regression Loss / Error Metrics
def Compute_Loss(Output_Pred, Output_Exact, Inputs, eps=1e-8, p=2):
    
    ## Loss for Primary Values 
    # Size of Batch
    B = Output_Exact.shape[0]

    # Mask (B, T, H, W)
    Mask = (Inputs[..., 0] != 0)

    # Flatten Batch
    Pred = Output_Pred.reshape(B, -1)
    Exact = Output_Exact.reshape(B, -1)
    Mask = Mask.reshape(B, -1)

    # Apply Mask
    Diff = (Pred - Exact) * Mask
    Exact = Exact * Mask

    # Per-Sample Norms
    Diff_Norm = torch.norm(Diff, p=p, dim=1)
    Exact_Norm = torch.norm(Exact, p=p, dim=1)

    # Relative Error Per Sample
    Loss = Diff_Norm / torch.clamp(Exact_Norm, min=1e-3)

    # First Derivative Loss with respect to radial distance
    Mask_dr = (Inputs[..., 0] != 0)[:, :, 1:, :]
    dr = Inputs[:, :, 1:, :, -3] - Inputs[:, :, :-1, :, -3]
    dPred  = Output_Pred[:, :, 1:, :] - Output_Pred[:, :, :-1, :]
    dExact = Output_Exact[:, :, 1:, :] - Output_Exact[:, :, :-1, :]
    dr = torch.where(Mask_dr, dr, torch.ones_like(dr))
    dPred  = (dPred  / dr) * Mask_dr
    dExact = (dExact / dr) * Mask_dr
    dDiff   = (dPred - dExact)
    Deriv_Loss_r = (torch.norm(dDiff.reshape(B, -1), p=p, dim=1) /
                torch.clamp(torch.norm(dExact.reshape(B, -1), p=p, dim=1), min=1e-3))
    
    # First Derivative Loss with respect to vertical direction (z)
    Mask_dz = (Inputs[..., 0] != 0)[:, 1:, :, :]
    dz = Inputs[:, 1:, :, :, -2] - Inputs[:, :-1, :, :, -2]
    dPred  = Output_Pred[:, 1:, :, :] - Output_Pred[:, :-1, :, :]
    dExact = Output_Exact[:, 1:, :, :] - Output_Exact[:, :-1, :, :]
    dz = torch.where(Mask_dz, dz, torch.ones_like(dz))
    dPred  = (dPred  / dz) * Mask_dz
    dExact = (dExact / dz) * Mask_dz
    dDiff = dPred - dExact
    Deriv_Loss_z = (torch.norm(dDiff.reshape(B, -1), p=p, dim=1) /
        torch.clamp(torch.norm(dExact.reshape(B, -1), p=p, dim=1), min=1e-3))

    # Batch Average
    return Loss.mean(), Deriv_Loss_r.mean(), Deriv_Loss_z.mean()

def Compute_MSE(Output_Pred, Output_True, Inputs):
    Mask = (Inputs[:,:,:,:,0]!= 0)
    MSE = (Output_Pred[Mask] - Output_True[Mask]) ** 2
    return MSE.mean()

def Compute_RMSE(Output_Pred, Output_True, Inputs):
    return torch.sqrt(Compute_MSE(Output_Pred, Output_True, Inputs))

def Compute_R2(y_pred, y_true, Inputs, eps=1e-8):
    Mask = (Inputs[:,:,:,:,0] != 0)

    yt = y_true[Mask].reshape(-1)
    yp = y_pred[Mask].reshape(-1)

    if yt.numel() == 0:
        return torch.tensor(0.0, device=y_true.device)

    yt_mean = yt.mean()
    ss_res = torch.sum((yt - yp) ** 2)
    ss_tot = torch.sum((yt - yt_mean) ** 2)

    # if variance is near zero, return 0
    if ss_tot < eps:
        return torch.tensor(0.0, device=y_true.device)

    r2 = 1.0 - ss_res / ss_tot
    return r2

def Compute_Regression_Metrics(y_pred, y_true, inputs):
    mse  = Compute_MSE(y_pred, y_true, inputs)
    rmse = torch.sqrt(mse)
    r2   = Compute_R2(y_pred, y_true, inputs)
    return mse, rmse, r2