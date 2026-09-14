## Functions/PostProcessing/Main.py

## Import Libraries 
from Functions.Import import *
from Functions.Training.Models.UNet3D import *
from Functions.Training.Models.CNN import *
from Functions.Training.Models.FNO import *
from Functions.Training.Models.UFNO import *
from Functions.Training.Models.VNet3D import *
from Functions.Training.PlotResults import *

def Run_Inference(Dataset, Model, Device, Dataset_Name, Result_Path, Params, Problem_Type, Plot_Results):
    MSE_Cum        = 0.0
    RMSE_Cum       = 0.0
    R2_Cum         = 0.0
    Loss_Cum       = 0.0
    Loss_Der_Cum_r = 0.0
    Loss_Der_Cum_z = 0.0
    Num_Batches    = 0

    with torch.no_grad():

        if Dataset_Name in ("Test", "Validation"):
            Iterator = tqdm(Dataset, desc=f"{Dataset_Name} Batches")
        elif Dataset_Name == "Train":
            BatchDir    = Dataset
            Batch_Files = sorted([f for f in os.listdir(BatchDir) if f.startswith(f'{Problem_Type}_Train_Batch_')])
            Iterator    = ({**torch.load(os.path.join(BatchDir, f), weights_only=False), "_file": f} for f in tqdm(Batch_Files, desc="Train Batches"))

        for Item in Iterator:
            if Dataset_Name == "Train":
                Input_Batch  = Item["Inputs"]
                Output_Batch = Item["Outputs"]
                Data_ID      = Item["Data_ID"]
            else:
                Input_Batch, Output_Batch, Data_ID = Item

            Input_Batch  = Input_Batch.to(Device,  dtype=torch.float32, non_blocking=True)
            Output_Batch = Output_Batch.to(Device, dtype=torch.float32, non_blocking=True)

            Output_Batch_Pred               = Model(Input_Batch)
            
            if Plot_Results is True:
                VisualizeResults(Output_Batch.detach().cpu(), Output_Batch_Pred.detach().cpu(), Input_Batch, Data_ID, Result_Path, Params, Problem_Type)
            
            # Unormalize and Compute Metrics
            Input_Batch, Output_Batch_Pred = DataUnnormalization(Inputs=Input_Batch.cpu(), Outputs=Output_Batch_Pred.detach().cpu(), Params=Params)
            Output_Batch = DataUnnormalization(Inputs=None, Outputs=Output_Batch.detach().detach().cpu(), Params=Params)

            Loss, Loss_Der_r, Loss_Der_z    = Compute_Loss(Output_Batch_Pred, Output_Batch, Input_Batch.detach().cpu())
            MSE_Batch, RMSE_Batch, R2_Batch = Compute_Regression_Metrics(Output_Batch_Pred, Output_Batch, Input_Batch.detach().cpu())

            MSE_Cum        += float(MSE_Batch)
            RMSE_Cum       += float(RMSE_Batch)
            R2_Cum         += float(R2_Batch)
            Loss_Cum       += Loss.item()
            Loss_Der_Cum_r += Loss_Der_r.item()
            Loss_Der_Cum_z += Loss_Der_z.item()
            Num_Batches    += 1

    MSE      = MSE_Cum        / Num_Batches
    RMSE     = RMSE_Cum       / Num_Batches
    R2       = R2_Cum         / Num_Batches
    Loss_Avg = Loss_Cum       / Num_Batches
    Loss_dr  = Loss_Der_Cum_r / Num_Batches
    Loss_dz  = Loss_Der_Cum_z / Num_Batches

    print(f"\n===== {Dataset_Name.upper()} RESULTS =====")
    print(f"  Loss     : {Loss_Avg:.4f}")
    print(f"  Loss dr  : {Loss_dr:.4f}")
    print(f"  Loss dz  : {Loss_dz:.4f}")
    print(f"  MSE      : {MSE:.4f}")
    print(f"  RMSE     : {RMSE:.4f}")
    print(f"  R²       : {R2:.4f}\n")

    return {"Loss": Loss_Avg, "Loss_dr": Loss_dr, "Loss_dz": Loss_dz,
            "MSE": MSE, "RMSE": RMSE, "R2": R2}


def Main(curDir, Model_Name, Problem_Type, Model_Type, Dataset_Config, Device):
    DataDir  = os.path.join(curDir, 'Dataset', Problem_Type)
    BatchDir = os.path.join(DataDir, 'Batched Data')

    # Batch Data if Toggled
    if Dataset_Config["Batch_Data"] is True:
        Train_Dataset = torch.load(os.path.join(DataDir, Problem_Type + '_Train.pt'), weights_only=False)
        os.makedirs(BatchDir, exist_ok=True)
        for i, (Inputs, Outputs, Data_ID) in enumerate(Train_Dataset):
            Batch = {"Inputs": Inputs, "Outputs": Outputs, "Data_ID": Data_ID}
            torch.save(Batch, os.path.join(BatchDir, f'{Problem_Type}_Train_Batch_{i:04d}.pt'))
        del Train_Dataset

    Test_Dataset  = torch.load(os.path.join(DataDir, Problem_Type + '_Test.pt'),  weights_only=False)
    Valid_Dataset = torch.load(os.path.join(DataDir, Problem_Type + '_Valid.pt'), weights_only=False)
    Params        = torch.load(os.path.join(DataDir, Problem_Type + '_Params.pt'), weights_only=False)

    # Load Model
    try:
        Model = torch.load(os.path.join(curDir, 'Saved Models', Problem_Type, Model_Type, Model_Name + '.pt'), weights_only=False)
        if isinstance(Model, torch.nn.DataParallel):
            Model = Model.module
        Model = Model.to(Device)

    except Exception:
        if Model_Type == "FNO":
            Model = FNO(12, 1, 10, 10, 10, 36, 128).to(Device)
        elif Model_Type == "U-FNO":
            Model = UFNO(12, 1, 10, 10, 10, 36, 128).to(Device)
        elif Model_Type == "U-Net":
            Model = UNet3D(12, 1, 32).to(Device)
        elif Model_Type == "V-Net":
            Model = VNet3D(12, 1, 32).to(Device)
        elif Model_Type == "CNN":
            Model = CNN(12, 1).to(Device)

        Checkpoint = torch.load(os.path.join(curDir, 'Saved Models', Problem_Type, Model_Type, Model_Name + '.pt'), weights_only=False)
        Model.load_state_dict(Checkpoint["Model_State"])

    Model.eval()

    if torch.cuda.device_count() > 1 and Model_Type not in ('FNO', 'U-FNO'):
        device_ids = list(range(torch.cuda.device_count()))
        Model = torch.nn.DataParallel(Model, device_ids=device_ids)

    # Print Torch Tensor Sizes
    Inputs, Outputs, Dataset_ID = next(iter(Test_Dataset))
    print(f"Input Shape:   {Inputs.shape}")
    print(f"Output Shape:  {Outputs.shape}")
    print(f"Dataset ID:    {Dataset_ID.shape}")
    print(f"Total Batches: {len(Test_Dataset)}")
    print(f"Total Samples: {len(Test_Dataset.dataset)}")

    # Torch Summary of Model
    _, Nr, Nz, Nt, Nc = Inputs.shape
    summary(Model, input_size=(Nr, Nz, Nt, Nc))

    Result_Path = os.path.join(curDir, 'Results', Problem_Type, Model_Type, Model_Name)
    os.makedirs(Result_Path, exist_ok=True)

    # Run Inference
    Test_Metrics  = Run_Inference(Test_Dataset, Model, Device, "Test",       Result_Path, Params, Problem_Type, Plot_Results=False)
    Valid_Metrics = Run_Inference(Valid_Dataset, Model, Device, "Validation", Result_Path, Params, Problem_Type, Plot_Results=False)
    Train_Metrics = Run_Inference(BatchDir,      Model, Device, "Train",      Result_Path, Params, Problem_Type, Plot_Results=False)

    # Save Metrics to CSV
    CSV_Path   = os.path.join(Result_Path, f"{Model_Name}_Metrics.csv")
    Fieldnames = ["Loss", "Loss (dL/dr)", "Loss (dz/dr)", "MSE", "RMSE", "R2"]

    Row = {
        "Loss"        : f"[{Train_Metrics['Loss']:.2E}, {Valid_Metrics['Loss']:.2E}, {Test_Metrics['Loss']:.2E}]",
        "Loss (dL/dr)": f"[{Train_Metrics['Loss_dr']:.2E}, {Valid_Metrics['Loss_dr']:.2E}, {Test_Metrics['Loss_dr']:.2E}]",
        "Loss (dz/dr)": f"[{Train_Metrics['Loss_dz']:.2E}, {Valid_Metrics['Loss_dz']:.2E}, {Test_Metrics['Loss_dz']:.2E}]",
        "MSE"         : f"[{Train_Metrics['MSE']:.2E}, {Valid_Metrics['MSE']:.2E}, {Test_Metrics['MSE']:.2E}]",
        "RMSE"        : f"[{Train_Metrics['RMSE']:.2E}, {Valid_Metrics['RMSE']:.2E}, {Test_Metrics['RMSE']:.2E}]",
        "R2"          : f"[{Train_Metrics['R2']:.3f}, {Valid_Metrics['R2']:.3f}, {Test_Metrics['R2']:.3f}]",
    }

    with open(CSV_Path, "w", newline="") as f:
        Writer = csv.DictWriter(f, fieldnames=Fieldnames)
        Writer.writeheader()
        Writer.writerow(Row)

    print(f"Metrics saved to: {CSV_Path}")

    return