# Functions/Preprocessing/DataPreprocessing.py

from Functions.Import import *

# Normalization Function
def DataNormalization(curDir, Inputs, Outputs, Save_Params, Norm_Dataset_Name):

    # Move Data to CPU
    Inputs = Inputs.to(device="cpu", dtype=torch.float32)
    Outputs = Outputs.to(device="cpu", dtype=torch.float32)
    Mask_Inputs = (Inputs != 0.0)
    Mask_Outputs = (Outputs != 0.0)

    # Replace -99 with NaN
    Inputs[~Mask_Inputs] = float('nan')
    Outputs[~Mask_Outputs] = float('nan')

    # Normalize Log Fields
    Inputs[..., 0:2][Mask_Inputs[..., 0:2]] = torch.log(Inputs[..., 0:2][Mask_Inputs[..., 0:2]])
    Inputs[..., 9][Mask_Inputs[..., 9]] = torch.log(Inputs[..., 9][Mask_Inputs[..., 9]]/ 99992.83)
    Inputs[..., 11][Mask_Inputs[..., 11]] = torch.log(Inputs[..., 11][Mask_Inputs[..., 11]]/ 10950.297)
    Outputs[Mask_Outputs] = torch.log10(Outputs[Mask_Outputs].clamp(min=1e-8))

    # Save Normalization Parameters
    if Save_Params:
        
        Params = {}     # Declare Normalization Paramater Structure

        # Convert to Numpy
        Inputs = Inputs.numpy()
        Outputs = Outputs.numpy()
        Mask_Inputs = Mask_Inputs.numpy()
        Mask_Outputs = Mask_Outputs.numpy()

        # Inputs Stats
        Inputs_Mean = numpy.nanmean(Inputs, axis=(0, 1, 2, 3))
        Inputs_Std  = numpy.nanstd(Inputs, axis=(0, 1, 2, 3))
        Inputs_Min = numpy.nanmin(Inputs, axis=(0, 1, 2, 3))
        Inputs_Max = numpy.nanmax(Inputs, axis=(0, 1, 2, 3))

        # Outputs stats
        Outputs_Mean = numpy.asarray(numpy.nanmean(Outputs[Mask_Outputs]), dtype=numpy.float32)
        Outputs_Std = numpy.asarray(numpy.nanstd(Outputs[Mask_Outputs]), dtype=numpy.float32)

        Params['Inputs']  = {'Mean': torch.from_numpy(Inputs_Mean), 'Std': torch.from_numpy(Inputs_Std),
                             'Min': torch.from_numpy(Inputs_Min), 'Max': torch.from_numpy(Inputs_Max)}
        Params['Outputs'] = {'Mean': torch.from_numpy(Outputs_Mean), 'Std': torch.from_numpy(Outputs_Std)}

        # Save Norm Parameters
        os.makedirs(os.path.join(curDir, 'Saved Models', Norm_Dataset_Name), exist_ok=True)
        torch.save(Params, os.path.join(curDir, 'Saved Models', Norm_Dataset_Name, 'Params.pt'))

        # Convert Back to Torch
        Inputs = torch.from_numpy(Inputs)
        Outputs = torch.from_numpy(Outputs)

    else:
        Params = torch.load(os.path.join(curDir, 'Saved Models', Norm_Dataset_Name, 'Params.pt'), weights_only=False)

    # Z-Score and Min-Max Scaling
    Inputs[..., 0:2].sub_(Params['Inputs']['Mean'][0:2]).div_(Params['Inputs']['Std'][0:2])
    Inputs[..., 3:9].sub_(Params['Inputs']['Min'][3:9]).div_(Params['Inputs']['Max'][3:9] - Params['Inputs']['Min'][3:9])
    Inputs[..., 9:].sub_(Params['Inputs']['Mean'][9:]).div_(Params['Inputs']['Std'][9:])
    Outputs.sub_(Params['Outputs']['Mean']).div_(Params['Outputs']['Std'])

    # Restore Mask
    Inputs[~Mask_Inputs] = 0
    Outputs[~Mask_Outputs] = 0

    return Inputs, Outputs, Params
    
def DataUnnormalization(Inputs, Outputs, Params):

    # Masks
    if Inputs is not None:
        Mask_Inputs  = (Inputs != 0)
    Mask_Outputs  = (Outputs != 0)

    # Undo Z-Score, Min-Max and Log Scaling
    if Inputs is not None:
        Inputs[..., 3:9] = Inputs[..., 3:9] * (Params['Inputs']['Max'][3:9] - Params['Inputs']['Min'][3:9])  + Params['Inputs']['Min'][3:9]
        Inputs[..., 0:2] = Inputs[..., 0:2] * Params['Inputs']['Std'][0:2] + Params['Inputs']['Mean'][0:2]
        Inputs[..., 0:2] = torch.exp(Inputs[..., 0:2])
        Inputs[..., 9][Mask_Inputs[..., 9]]  = 99992.83 * torch.exp(Inputs[..., 9][Mask_Inputs[..., 9]])
        Inputs[..., 11][Mask_Inputs[..., 11]] = 10950.297 * torch.exp(Inputs[..., 11][Mask_Inputs[..., 11]])
        Inputs[..., 9:] = Inputs[..., 9:] * Params['Inputs']['Std'][9:] + Params['Inputs']['Mean'][9:]
    Outputs = 10 ** (Outputs * Params['Outputs']['Std'] + Params['Outputs']['Mean'])
    
    # Restore Mask
    if Inputs is not None:
        Inputs[~Mask_Inputs] = 0
    Outputs[~Mask_Outputs] = 0

    if Inputs is None:
        return Outputs
    else: 
        return Inputs, Outputs

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, Inputs, Outputs, Dataset_ID):
        self.Inputs = Inputs
        self.Outputs = Outputs
        self.Dataset_ID = Dataset_ID

    def __len__(self):
        return self.Inputs.size(0)

    def __getitem__(self, idx):
        return self.Inputs[idx], self.Outputs[idx], self.Dataset_ID[idx]