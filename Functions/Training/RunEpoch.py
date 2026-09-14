## Functions/Training/Main.py

## Import Libraries 
from Functions.Import import *
from Functions.Training.ComputeErrors import Compute_Regression_Metrics, Compute_MSE

def RunEpoch(Dataset, Model, Device, Optimizer, Scheduler, Loss, Train_Toggle):

    Num_Batches = 0
    MSE_Cum = 0.0
    RMSE_Cum = 0.0
    R2_Cum = 0.0

    if Train_Toggle:
        Model.train()
    else:
        Model.eval()

    for Input_Batch, Output_Batch, ____ in Dataset:

        Input_Batch = Input_Batch.to(Device, dtype=torch.float32)
        Output_Batch = Output_Batch.to(Device, dtype=torch.float32)

        if Train_Toggle:
            Optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(Train_Toggle):
            Output_Batch_Pred = Model(Input_Batch)
            Loss_MSE = Compute_MSE(Output_Batch_Pred, Output_Batch, Input_Batch)

            if Train_Toggle:
                Loss_MSE.backward()
                Optimizer.step()

        Model.eval()
        MSE_Batch, RMSE_Batch, R2_Batch = Compute_Regression_Metrics(Output_Batch_Pred.detach(), Output_Batch, Input_Batch)
        MSE_Cum += float(MSE_Batch)
        RMSE_Cum += float(RMSE_Batch)
        R2_Cum += float(R2_Batch)

        Num_Batches += 1

    Avg_MSE = MSE_Cum / Num_Batches
    Avg_RMSE = RMSE_Cum / Num_Batches
    Avg_R2 = R2_Cum / Num_Batches

    return Avg_MSE, Avg_RMSE, Avg_R2