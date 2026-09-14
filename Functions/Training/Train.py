## Functions/Training/Train.py

## Import Libraries 
from Functions.Import import *
from Functions.Training.PlotResults import *
from Functions.Training.ComputeErrors import *

def Train(Model, Optimizer, Scheduler, Train_Dataset, Validation_Dataset, Model_Config, Toggle_Config, Device, Resume_From):

    Lambda_r     = Model_Config["Lambda_r"]
    Lambda_z     = Model_Config["Lambda_z"]
    N_Epoch      = Model_Config["N_Epoch"]
    Epoch_Tensor = torch.zeros(N_Epoch, 1)
    MSE_Tensor   = torch.zeros(N_Epoch, 2)
    RMSE_Tensor  = torch.zeros(N_Epoch, 2)
    R2_Tensor    = torch.zeros(N_Epoch, 2)
    Loss_Tensor  = torch.zeros(N_Epoch, 6)

    Start_Epoch = 0
    max_R2     = -float("inf")

    ## Resume from Checkpoint
    if Resume_From is not None and os.path.exists(Resume_From):
        print(f"Resuming from checkpoint: {Resume_From}")
        Checkpoint = torch.load(Resume_From, weights_only=False)

        Model.load_state_dict(Checkpoint["Model_State"])
        Optimizer.load_state_dict(Checkpoint["Optimizer_State"])
        if Scheduler is not None and Checkpoint.get("Scheduler_State") is not None:
            Scheduler.load_state_dict(Checkpoint["Scheduler_State"])

        Start_Epoch  = Checkpoint["Epoch"]          
        max_R2      = Checkpoint["max_R2"]

        # Restore Metric History 
        completed = Start_Epoch 
        Epoch_Tensor[:completed] = Checkpoint["Epoch_Tensor"][:completed]
        MSE_Tensor[:completed]   = Checkpoint["MSE_Tensor"][:completed]
        RMSE_Tensor[:completed]  = Checkpoint["RMSE_Tensor"][:completed]
        Loss_Tensor[:completed]  = Checkpoint["Loss_Tensor"][:completed]
        R2_Tensor[:completed]    = Checkpoint["R2_Tensor"][:completed]

        print(f"  Resumed at epoch {Start_Epoch + 1} / {N_Epoch}  |  Best R2 so far: {max_R2:.4e}")

    Model = Model.to(Device)
    pbar  = tqdm(range(Start_Epoch, N_Epoch), desc="Progress", unit="Epoch", dynamic_ncols=True, leave=True)

    for epoch in pbar:
        # Training
        Model.train()
        MSE_Cum = RMSE_Cum = R2_Cum = Loss_Cum = Loss_Der_Cum_r = Loss_Der_Cum_z = 0.0
        Num_Batches = 0

        for Input_Batch, Output_Batch, Dataset_ID_Batch in Train_Dataset:
            Input_Batch  = Input_Batch.to(Device,  dtype=torch.float32, non_blocking=True)
            Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            Optimizer.zero_grad(set_to_none=True)
            Output_Batch_Pred            = Model(Input_Batch)
            Loss, Loss_Der_r, Loss_Der_z = Compute_Loss(Output_Batch_Pred, Output_Batch, Input_Batch)
            Total_Loss                   = Loss + Lambda_r * Loss_Der_r + Lambda_z * Loss_Der_z
            Total_Loss.backward()
            Optimizer.step()

            MSE_Batch, RMSE_Batch, R2_Batch = Compute_Regression_Metrics(Output_Batch_Pred.detach(), Output_Batch, Input_Batch)
            MSE_Cum        += float(MSE_Batch)
            RMSE_Cum       += float(RMSE_Batch)
            R2_Cum         += float(R2_Batch)
            Loss_Cum       += Loss.item()
            Loss_Der_Cum_r += Loss_Der_r.item()
            Loss_Der_Cum_z += Loss_Der_z.item()
            Num_Batches    += 1

        Epoch_Tensor[epoch, 0] = epoch + 1
        MSE_Tensor[epoch, 0]   = MSE_Cum        / Num_Batches
        RMSE_Tensor[epoch, 0]  = RMSE_Cum       / Num_Batches
        R2_Tensor[epoch, 0]    = R2_Cum         / Num_Batches
        Loss_Tensor[epoch, 0]  = Loss_Cum       / Num_Batches
        Loss_Tensor[epoch, 1]  = Loss_Der_Cum_r / Num_Batches
        Loss_Tensor[epoch, 2]  = Loss_Der_Cum_z / Num_Batches

        # Validation
        Model.eval()
        MSE_Cum = RMSE_Cum = R2_Cum = Loss_Cum = Loss_Der_Cum_r = Loss_Der_Cum_z = 0.0
        Num_Batches = 0

        with torch.inference_mode():
            for Input_Batch, Output_Batch, Dataset_ID_Batch in Validation_Dataset:
                Input_Batch  = Input_Batch.to(Device,  dtype=torch.float32, non_blocking=True)
                Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)

                Output_Batch_Pred            = Model(Input_Batch)
                Loss, Loss_Der_r, Loss_Der_z = Compute_Loss(Output_Batch_Pred, Output_Batch, Input_Batch)

                MSE_Batch, RMSE_Batch, R2_Batch = Compute_Regression_Metrics(Output_Batch_Pred, Output_Batch, Input_Batch)
                MSE_Cum        += float(MSE_Batch)
                RMSE_Cum       += float(RMSE_Batch)
                R2_Cum         += float(R2_Batch)
                Loss_Cum       += Loss.item()
                Loss_Der_Cum_r += Loss_Der_r.item()
                Loss_Der_Cum_z += Loss_Der_z.item()
                Num_Batches    += 1

        MSE_Tensor[epoch, 1]  = MSE_Cum        / Num_Batches
        RMSE_Tensor[epoch, 1] = RMSE_Cum       / Num_Batches
        R2_Tensor[epoch, 1]   = R2_Cum         / Num_Batches
        Loss_Tensor[epoch, 3] = Loss_Cum       / Num_Batches
        Loss_Tensor[epoch, 4] = Loss_Der_Cum_r / Num_Batches
        Loss_Tensor[epoch, 5] = Loss_Der_Cum_z / Num_Batches

        if Scheduler is not None:
            Scheduler.step()

        # Visualize
        if Toggle_Config["Visual_Results"] and epoch > 0 and epoch % Toggle_Config["Plot_Rate"] == 0:
            Result_Path = os.path.join(Model_Config["Result_Path"], f"Epoch {epoch}")
            os.makedirs(Result_Path, exist_ok=True)
            VisualizeResults(Output_Batch, Output_Batch_Pred, Input_Batch, Dataset_ID_Batch, Result_Path, Model_Config["Params"], Model_Config["Problem_Type"])

        # Save Best Model
        if R2_Tensor[epoch, 1] > max_R2:
            max_R2 = R2_Tensor[epoch, 1]
            torch.save(Model, os.path.join(Model_Config["Model_Path"], f'{Model_Config["Model_Name"]}.pt'))

        if (epoch + 1) % 10 == 0:
            Checkpoint = {
                "Epoch"           : epoch + 1,          
                "max_R2"         : max_R2,
                "Model_State"     : Model.state_dict(),
                "Optimizer_State" : Optimizer.state_dict(),
                "Scheduler_State" : Scheduler.state_dict() if Scheduler is not None else None,
                "Epoch_Tensor"    : Epoch_Tensor,
                "MSE_Tensor"      : MSE_Tensor,
                "RMSE_Tensor"     : RMSE_Tensor,
                "R2_Tensor"       : R2_Tensor,
                "Loss_Tensor"     : Loss_Tensor,
            }
            torch.save(Checkpoint, os.path.join(Model_Config["Model_Path"], f'{Model_Config["Model_Name"]}_Epoch_{epoch+1}.pt'))

        pbar.set_postfix(Loss    = f"[{Loss_Tensor[epoch, 0]:.4e}, {Loss_Tensor[epoch, 3]:.4e}]",
                         Loss_dr = f"[{Loss_Tensor[epoch, 1]:.4e}, {Loss_Tensor[epoch, 4]:.4e}]",
                         Loss_dz = f"[{Loss_Tensor[epoch, 2]:.4e}, {Loss_Tensor[epoch, 5]:.4e}]",
                         R2      = f"[{R2_Tensor[epoch, 0]:.4e}, {R2_Tensor[epoch, 1]:.4e}]",)

    # Post-Training 
    PlotTrainingMetrics(Epoch_Tensor, MSE_Tensor, RMSE_Tensor, R2_Tensor, Model_Config["Result_Path"], Model_Config["Model_Name"])

    if Toggle_Config["Visual_Results"]:
        Result_Path = os.path.join(Model_Config["Result_Path"], f"Epoch {N_Epoch}")
        os.makedirs(Result_Path, exist_ok=True)
        VisualizeResults(Output_Batch, Output_Batch_Pred, Input_Batch, Dataset_ID_Batch, Result_Path, Model_Config["Params"], Model_Config["Problem_Type"])

    Results_Struc = {"Loss": Loss_Tensor.cpu().numpy(), "MSE": MSE_Tensor.cpu().numpy(),
                     "RMSE": RMSE_Tensor.cpu().numpy(), "R2":  R2_Tensor.cpu().numpy(),
                     "Epoch": Epoch_Tensor.cpu().numpy()}
    scipy.io.savemat(os.path.join(Model_Config["Result_Path"], f'{Model_Config["Model_Name"]}_Results.mat'), Results_Struc)