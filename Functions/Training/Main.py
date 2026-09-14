## Functions/Training/Main.py

## Import Libraries 
from Functions.Import import *
from Functions.Training.Models.UNet3D import *
from Functions.Training.Models.CNN import *
from Functions.Training.Models.FNO import *
from Functions.Training.Models.UFNO import *
from Functions.Training.Models.VNet3D import *
from Functions.Training.Train import *
from itertools import islice

## Define Training Function
def Main(curDir, Model_Config, Dataset_Config, Optimizer_Config, Scheduler_Config, Toggle_Config, Device):

    # Load Dataset
    Train_Dataset_Raw = torch.load(os.path.join(curDir, 'Dataset', Model_Config["Problem_Type"], Model_Config["Problem_Type"] + '_Train.pt'), weights_only=False)
    Valid_Dataset_Raw = torch.load(os.path.join(curDir, 'Dataset', Model_Config["Problem_Type"], Model_Config["Problem_Type"] + '_Valid.pt'), weights_only=False)
    Model_Config["Params"] = torch.load(os.path.join(curDir, 'Dataset', Model_Config["Problem_Type"], Model_Config["Problem_Type"] + '_Params.pt'), weights_only=False)

    Train_Dataset = torch.utils.data.DataLoader(
    Train_Dataset_Raw.dataset,
    batch_size  = Dataset_Config['Batch_Size'],          
    shuffle     = True,
    num_workers = Train_Dataset_Raw.num_workers,
    pin_memory  = Train_Dataset_Raw.pin_memory,
)
    Valid_Dataset = torch.utils.data.DataLoader(
    Valid_Dataset_Raw.dataset,
    batch_size  = Dataset_Config['Batch_Size'],
    shuffle     = False,
    num_workers = Valid_Dataset_Raw.num_workers,
    pin_memory  = Valid_Dataset_Raw.pin_memory,
)

    ## Create Model Path Name
    Base_Model_Path  = os.path.join(curDir, 'Saved Models',        f'{Model_Config["Model_Name"]}')
    Base_Result_Path = os.path.join(curDir, 'Performance Metrics', f'{Model_Config["Model_Name"]}')
    Model_Config["Result_Path"] = Base_Result_Path
    Model_Config["Model_Path"]  = Base_Model_Path
    os.makedirs(Base_Model_Path,  exist_ok=True)
    os.makedirs(Base_Result_Path, exist_ok=True)

    # Auto-detect latest resumable checkpoint
    Checkpoint_Files = sorted([
        f for f in os.listdir(Base_Model_Path)
        if f.startswith(Model_Config["Model_Name"]) and "_Epoch_" in f and f.endswith(".pt")
    ], key=lambda f: int(f.split("_Epoch_")[-1].replace(".pt", "")))
    Resume_From = os.path.join(Base_Model_Path, Checkpoint_Files[-1]) if Checkpoint_Files else None

    # Initialize Model
    if Model_Config["Model_Type"] == "FNO":
        Model = FNO(Model_Config["In_Channels"],
                    Model_Config["Out_Channels"],
                    Model_Config["Modes_1"],
                    Model_Config["Modes_2"],
                    Model_Config["Modes_3"],
                    Model_Config["Width"],
                    Model_Config["dh"]).to(Device)

    elif Model_Config["Model_Type"] == "U-FNO":
        Model = UFNO(Model_Config["In_Channels"],
                     Model_Config["Out_Channels"],
                     Model_Config["Modes_1"],
                     Model_Config["Modes_2"],
                     Model_Config["Modes_3"],
                     Model_Config["Width"],
                     Model_Config["dh"]).to(Device)

    elif Model_Config["Model_Type"] == "U-Net":
        Model = UNet3D(Model_Config["In_Channels"],
                       Model_Config["Out_Channels"],
                       Model_Config["Base_Channels"]).to(Device)

    elif Model_Config["Model_Type"] == "V-Net":
        Model = VNet3D(Model_Config["In_Channels"],
                       Model_Config["Out_Channels"],
                       Model_Config["Base_Channels"]).to(Device)

    elif Model_Config["Model_Type"] == "CNN":
        Model = CNN(Model_Config["In_Channels"],
                    Model_Config["Out_Channels"]).to(Device)

    # Initialize Optimizer & Scheduler
    Optimizer = torch.optim.Adam(
        Model.parameters(),
        lr           = Optimizer_Config["Learning_Rate"],
        betas        = Optimizer_Config["Betas"],
        eps          = Optimizer_Config["Eps"],
        weight_decay = Optimizer_Config["Weight_Decay"]
    )

    Scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        Optimizer,
        T_0    = Scheduler_Config["T_0"],
        T_mult = Scheduler_Config["T_mult"],
        eta_min= Scheduler_Config["Eta_Min"]
    )

    # Load Checkpoint States
    if Resume_From:
        print(f"Loading checkpoint: {os.path.basename(Resume_From)}")
        Checkpoint = torch.load(Resume_From, weights_only=False)
        Model.load_state_dict(Checkpoint["Model_State"])
        Optimizer.load_state_dict(Checkpoint["Optimizer_State"])
        if Checkpoint.get("Scheduler_State") is not None:
            Scheduler.load_state_dict(Checkpoint["Scheduler_State"])

    if Resume_From:
        print(f"\nFound checkpoint — resuming from: {os.path.basename(Resume_From)}")
    else:
        print("\nNo checkpoint found — training from scratch.")

        # Print Model
        print("\n===== MODEL SUMMARY =====\n")
        print(Model)
        print("Number of Learnable Parameters:", f"{sum(p.numel() for p in Model.parameters() if p.requires_grad):,}")

    # Train Model
    Train(Model, Optimizer, Scheduler, Train_Dataset, Valid_Dataset,
          Model_Config, Toggle_Config, Device, Resume_From=Resume_From)

    return