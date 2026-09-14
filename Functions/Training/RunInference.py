# Functions/Training/RunInference.py

# Import Libraries 
from Functions.Import import *
from Functions.Training.VisualizeResults import *

def RunInference(Dataset, Model, Device, Result_Path, Params):

    ## Run through Batches
    for Input_Batch, Output_Batch_True, Dataset_ID_Batch in Dataset:

        # Move Batch to Device
        Input_Batch = Input_Batch.to(Device, dtype=torch.float32)

        # Forward Pass 
        with torch.no_grad():
            Output_Batch_Pred = Model(Input_Batch)

        # Repermute Inputs and Outputs
        Output_Batch_Pred = Output_Batch_Pred.detach().cpu()
        Output_Batch_True = Output_Batch_True.cpu()
        Input_Batch = Input_Batch.cpu()

        # Visualize Results
        VisualizeResults(Output_Batch_True, Output_Batch_Pred, Input_Batch, Dataset_ID_Batch, Result_Path, Params)

    return