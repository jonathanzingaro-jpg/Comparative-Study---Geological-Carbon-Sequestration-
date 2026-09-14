# Functions/Preprocessing/Main.py

# Import Libraries
from Functions.Import import *

# Main Function
def Main(curDir, Norm_Type_Inputs, Norm_Type_Outputs, Data_Split, Norm_Dataset_Name, Dataset_Name, Batch_Size, Save_Params, Terminal_Toggle):

    ## Load Raw Dataset
    Inputs = torch.load(os.path.join(curDir, 'Dataset', Dataset_Name[0]), weights_only=True, mmap=True)
    Outputs = torch.load(os.path.join(curDir, 'Dataset', Dataset_Name[1]), weights_only=True, mmap=True)

    # Modify Inputs Mask
    Mask = Inputs[:, :, :, :, 0] == 0.0
    Inputs[Mask] = -99
    Outputs = (Outputs[:, :, :, -2:-1])
    Inputs = (Inputs[:, :, :, -2:-1,:])

    ## Make Directory
    os.makedirs(os.path.join(curDir, 'Dataset', f'{Norm_Dataset_Name}'), exist_ok=True)

    ## Print Size of Tensors
    if Terminal_Toggle is True:
        print(f'\nShape of Inputs: {Inputs.shape}, Shape of Outputs: {Outputs.shape}')

    ## Perform Normalization 
    #Inputs, Outputs, ____ = DataNormalization(curDir, Inputs, Outputs, Norm_Type_Inputs, Norm_Type_Outputs, Save_Params, Norm_Dataset_Name, Terminal_Toggle)

    # Declare Dataset using Dataloader
    Dataset_ID = torch.arange(1, Inputs.shape[0] + 1, 1) 
    Dataset = CustomDataset(Inputs, Outputs, Dataset_ID)
    Train_Dataset, Validation_Dataset, Test_Dataset = torch.utils.data.random_split(Dataset, (Data_Split[0], Data_Split[1], Data_Split[2]))

    # Training Dataset
    Train_Dataset = DataLoader(dataset=Train_Dataset, batch_size=Batch_Size, shuffle=True, drop_last=False)

    # Validation Dataset
    Validation_Dataset = DataLoader(dataset=Validation_Dataset, batch_size=Batch_Size, shuffle=True, drop_last=False)

    # Testing Dataset
    Test_Dataset = DataLoader(dataset=Test_Dataset, batch_size=Batch_Size, shuffle=True, drop_last=False)

    ## Print Message
    if Terminal_Toggle is True:
        print(f'\nSaving Datasets...')

    # Save Datasets
    torch.save(Train_Dataset, os.path.join(curDir, 'Dataset', f'{Norm_Dataset_Name}', 'Train_Dataset.pth'))
    torch.save(Validation_Dataset, os.path.join(curDir, 'Dataset', f'{Norm_Dataset_Name}', 'Validation_Dataset.pth'))
    torch.save(Test_Dataset, os.path.join(curDir, 'Dataset', f'{Norm_Dataset_Name}', 'Test_Dataset.pth'))
    
    if Terminal_Toggle is True:
        print(f'\nComplete!')

    return