## Functions/Training/InitializeModel.py

## Import Libraries 
from Functions.Import import *
from Functions.Training.Models.UNet3D import UNet3D
from Functions.Training.Models.CustomModel import CustomCNN


def InitializeModel(Model_Config, Optimizer_Config, Scheduler_Config, Device):
    """
    Initializes the model, optimizer, and LR scheduler.
    Returns:
        (Model, Optimizer, Scheduler)
    """

    ## Intialize Model
    # Check if model type is supported
    Supported_Models = ['pre-trained-U-Net', 'UNet3D', 'Development Model']
    if Model_Config['Model_Type'] not in Supported_Models:
        raise ValueError(f"Unsupported model type '{Model_Config["Model_Type"]}'. "f"Supported options are: {Supported_Models}")
    
    required_inputs = ['Model_Type', 'In_Channels', 'Out_Channels'] # Required Inputs for Models
    for input in required_inputs:
        if input not in Model_Config:
            raise ValueError(f"Missing required '{input}' in Model_Config for model initialization.")

    # Initialize Model
    if Model_Config['Model_Type'] == "Development Model":
        Model = CustomCNN(in_channels=Model_Config['In_Channels'], out_channels=Model_Config['Out_Channels'])

    elif Model_Config['Model_Type'] == "UNet3D":
        Model = UNet3D(in_channels=Model_Config['In_Channels'], out_channels=Model_Config['Out_Channels'])
    
    else:
        raise ValueError(f"Unknown Model_Type: {Model_Config['Model_Type']}")

    # Move Model to Device
    Model = Model.to(Device, dtype=torch.float32)

    ## Initialize Optimizer 
    # Check if optimizer type is supported 
    Supported_Optimizers = ['Adam']
    if Optimizer_Config["Optimizer_Type"] not in Supported_Optimizers:
        raise ValueError(f"Unsupported optimizer type '{Optimizer_Config['Optimizer_Type']}'. "f"Supported options are: {Supported_Optimizers}")

    # Initialize Optimizer
    if Optimizer_Config["Optimizer_Type"] == "Adam":

        # Raise Error for Inputs 
        required_inputs = ["Learning_Rate", "Weight_Decay", "Betas", "Eps"] # Required Inputs for Adam
        for input in required_inputs:
            if input not in Optimizer_Config:
                raise ValueError(f"Missing required '{input}' in Optimizer_Config for Adam optimizer.")

        # Define Optimizer 
        Optimizer = torch.optim.Adam(Model.parameters(), 
                                     lr=Optimizer_Config["Learning_Rate"], 
                                     betas=Optimizer_Config["Betas"], 
                                     eps=Optimizer_Config["Eps"], 
                                     weight_decay=Optimizer_Config["Weight_Decay"])
        
    ## Initialize Scheduler
    Supported_Schedulers = ['CosineAnnealingLR', 'None']
    if Scheduler_Config["Scheduler_Type"] not in Supported_Schedulers:
        raise ValueError(f"Unsupported scheduler type '{Scheduler_Config['Scheduler_Type']}'. "f"Supported options are: {Supported_Schedulers}")

    if Scheduler_Config["Scheduler_Type"] == "CosineAnnealingLR":
        required_inputs = ["T_max", "Min_Learning_Rate"] # Required Inputs for CosineAnnealingLR
        for key in required_inputs:
            if key not in Scheduler_Config:
                raise ValueError(f"Missing required '{key}' in Scheduler_Config for CosineAnnealingLR scheduler.")
    
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=Scheduler_Config["T_max"], eta_min=Scheduler_Config["Min_Learning_Rate"])

    elif Scheduler_Config["Scheduler_Type"] == "None":
        Scheduler = None

    return Model, Optimizer, Scheduler

def InitializeOptimizer(Optimizer_Config, Model):

    ## Initialize Optimizer 
    # Check if optimizer type is supported 
    Supported_Optimizers = ['Adam']
    if Optimizer_Config["Optimizer_Type"] not in Supported_Optimizers:
        raise ValueError(f"Unsupported optimizer type '{Optimizer_Config['Optimizer_Type']}'. "f"Supported options are: {Supported_Optimizers}")

    # Initialize Optimizer
    if Optimizer_Config["Optimizer_Type"] == "Adam":

        # Raise Error for Inputs 
        required_inputs = ["Learning_Rate", "Weight_Decay", "Betas", "Eps"] # Required Inputs for Adam
        for input in required_inputs:
            if input not in Optimizer_Config:
                raise ValueError(f"Missing required '{input}' in Optimizer_Config for Adam optimizer.")

        # Define Optimizer 
        Optimizer = torch.optim.Adam(Model.parameters(), 
                                     lr=Optimizer_Config["Learning_Rate"], 
                                     betas=Optimizer_Config["Betas"], 
                                     eps=Optimizer_Config["Eps"], 
                                     weight_decay=Optimizer_Config["Weight_Decay"])
    return Optimizer

