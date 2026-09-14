# Functions/Visualization/Main.py

# Import Libraries 
from Functions.Import import *

# Main Function
def Main(curDir, DataDir, DatasetName, Visualize_Toggle, Rewrite_Toggle, Save_Data, Problem_Type, OutputDatasetName):

    ## Import Dataset 
    Inputs = torch.load(os.path.join(DataDir, Problem_Type, DatasetName[0]), weights_only=True)
    Outputs = torch.load(os.path.join(DataDir, Problem_Type, DatasetName[1]), weights_only=True)

    # Print Data Metrics 
    print(f'Mean: {torch.mean(Outputs)}, Std: {torch.std(Outputs)}')  
    print(f'Min: {torch.min(Outputs)}, Max: {torch.max(Outputs)}')   

    # Loop over Datasets
    n = Inputs.shape[0]     # Number of Datasets
    pbar = tqdm(range(n), desc="Datasets", unit="dataset") 
    for i in pbar:
        
        Dataset_ID = i + 1
        
        # Plotting Functions
        Inputs[i, ...] = VisualizeInputs(Inputs[i,...], curDir, Dataset_ID, Visualize_Toggle, Rewrite_Toggle, Problem_Type)                                       # Plot Inputs for Dataset and Unnormalize Dataset
        Outputs[i, ...] = VisualizeOutputs(Inputs[i,...], Outputs[i,...], curDir, Dataset_ID, Visualize_Toggle, Problem_Type)                                     # Plot Inputs for Dataset and Unnormalize Dataset

        pbar.update(0) # Update Progress Bar

    # Print Data Metrics 
    Mask = Outputs != -99
    print(f'Mean: {torch.mean(Outputs[Mask])}, Std: {torch.std(Outputs[Mask])}')   

    if Save_Data is True:
    
        # Save Dataset
        os.makedirs(os.path.join(curDir, 'Dataset'), exist_ok=True)
        torch.save(Inputs, os.path.join(curDir, 'Dataset', OutputDatasetName[0]))
        torch.save(Outputs, os.path.join(curDir, 'Dataset', OutputDatasetName[1]))

        # Print Size of Inputs and Outputs
        print(f'Shape of Inputs: {Inputs.shape}')
        print(f'Shape of Outputs: {Outputs.shape}')

    return

