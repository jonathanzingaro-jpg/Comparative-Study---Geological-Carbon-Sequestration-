# Functions/Training/VisualizeData.py

# Import Libraries 
from Functions.Import import *
from Functions.Preprocessing import *
import multiprocessing

# Visualize Input Function
def Visualize_Single_Dataset(Output_True, Output_Pred, Input, Dataset_ID, Result_Path, X, Y, times, Problem_Type):

    # Define Mask and Thickness
    Mask = Input[:,:,0,0] != 0
    Thickness = sum(Mask[:,0])
    
    # Extract Input Parameters
    Phi_Map = Input[:,:,0,2][Mask].reshape((Thickness, -1))
    kr_Map  = Input[:,:,0,0][Mask].reshape((Thickness, -1))
    kz_Map  = Input[:,:,0,1][Mask].reshape((Thickness, -1))
    Inj_Rate = Input[0,0,0,4]
    Temp     = Input[0,0,0,5]
    Pressure = Input[0,0,0,6]
    Swi      = Input[0,0,0,7]
    lam      = Input[0,0,0,8]

    # Save Inputs and Outputs to .mat File
    Output_Data = {"kr_Map":  Input[:,:,0,0],
               "kz_Map":  Input[:,:,0,1],
               "Phi_Map":  Input[:,:,0,2],
               "Inj_Rate": Inj_Rate,
               "Temp":     Temp,
               "Pressure": Pressure,
               "Swi":      Swi,
               "Lambda":   lam,
               "Output_True": Output_True,
               "Output_Pred": Output_Pred,
               "R": Input[:,:,0,-3],
               "Z": Input[:,:,0,-2]}
    os.makedirs(os.path.join(Result_Path, f'Dataset {Dataset_ID}'), exist_ok=True)
    savemat(os.path.join(Result_Path, f'Dataset {Dataset_ID}', f'Dataset {Dataset_ID}' + "_Outputted_Results.mat"), Output_Data)


    t_lst = [14,20,23]                                       # Time Indices to Plot
    plt.figure(figsize=(20,9), constrained_layout=True)      # Define Figure   

    Save_Path = os.path.join(Result_Path, f'Dataset {Dataset_ID}')
    os.makedirs(Save_Path, exist_ok=True)

    for j, t in enumerate(t_lst):       
        plt.subplot(4,3,j+1)
        
        if j == 2:
            plt.jet()
            plt.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(Phi_Map), shading='auto')
            plt.title(r'$\phi$ (-)')
        elif j == 1:
            plt.jet()
            plt.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(kz_Map), shading='auto')
            plt.title(r'$k_z$ (mD)')
        else:
            plt.jet()
            plt.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(kr_Map), shading='auto')
            plt.title(r'$k_r$ (mD)')
        plt.colorbar(fraction=0.02)
        plt.xlim([0,3500])      

        # Compute Limits 
        if Problem_Type == 'dP':
            vmin = min(numpy.min(Output_True[:,:,t][Mask].reshape((Thickness, -1))), numpy.min(Output_Pred[:,:,t][Mask].reshape((Thickness, -1))))
            vmax = max(numpy.max(Output_True[:,:,t][Mask].reshape((Thickness, -1))), numpy.max(Output_Pred[:,:,t][Mask].reshape((Thickness, -1))))  
        elif Problem_Type == 'SG': 
            vmin = 0
            vmax = 1.0

        # Exact Solution
        plt.subplot(4,3,j+4)
        plt.jet()
        plt.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(Output_True[:,:,t][Mask].reshape((Thickness, -1))), shading='auto', vmin=vmin, vmax=vmax)

        if Problem_Type == 'dP':
            plt.title(r'$dP$ (bar), '+f't={times[t]:.1f} yr')
        elif Problem_Type == 'SG':
            plt.title(r'$SG$ (-), '+f't={times[t]:.1f} yr')

        plt.colorbar(fraction=0.02)
        plt.xlim([0,3500])

        # Predicted Solution
        plt.subplot(4,3,j+7)
        plt.jet()
        plt.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(Output_Pred[:,:,t][Mask].reshape((Thickness, -1))), shading='auto', vmin=vmin, vmax=vmax)

        if Problem_Type == 'dP':
            plt.title(r'$\hat{dP}$ (bar), '+f't={times[t]:.1f} yr')
        elif Problem_Type == 'SG':
            plt.title(r'$\hat{SG}$ (-), '+f't={times[t]:.1f} yr')

        plt.colorbar(fraction=0.02)
        plt.xlim([0,3500])

        # Error 
        Error = numpy.abs((Output_Pred[:,:,t][Mask].reshape((Thickness, -1)) - Output_True[:,:,t][Mask].reshape((Thickness, -1)))) ** 2 / numpy.max(numpy.abs(Output_True[:,:,t][Mask].reshape((Thickness, -1)))) ** 2

        plt.subplot(4,3,j+10)

        # Define xlim range
        x_min, x_max = 0, 3500

        # Create mask for the xlim range
        x_mask = (X[:Thickness, :] >= x_min) & (X[:Thickness, :] <= x_max)

        # Apply mask to get only the visible data
        X_clipped = X[:Thickness, :][x_mask].reshape((Thickness, -1))
        Y_clipped = Y[:Thickness, :][x_mask].reshape((Thickness, -1))
        Error_clipped = numpy.flipud(Error)[x_mask[::-1, :]].reshape((Thickness, -1))

        plt.pcolor(X_clipped, Y_clipped, Error_clipped, 
           shading='auto', norm=LogNorm(), cmap='jet')
        plt.colorbar(fraction=0.02)

        if Problem_Type == 'dP':
            plt.title(r'$|dP-\hat{dP}|_2 /|dP|_2$ (-), ' + f't={times[t]:.1f} yr')
        elif Problem_Type == 'SG':
            plt.title(r'$|SG-\hat{SG}|_2 /|SG|_2$ (-), ' + f't={times[t]:.1f} yr')
        plt.xlim([x_min, x_max])
    
    plt.suptitle(f'Params: Injection Rate: {Inj_Rate:.2f} MT/yr, Temperature: {Temp:.1f} °C, Initial Pressure: {Pressure:.1f} bar, Swi: {Swi:.2f}, Lambda: {lam:.2f}')

    if Problem_Type == 'dP':
        plt.savefig(os.path.join(Save_Path, f'dP_Results'), dpi=600)
    elif Problem_Type == 'SG':
        plt.savefig(os.path.join(Save_Path, f'SG_Results'), dpi=600)

    plt.close()


def VisualizeResults(Output_True_Batch, Output_Pred_Batch, Input_Batch, Dataset_ID_Batch, Result_Path, Params, Problem_Type):

    # Define Spatial and Temporal Values
    dx = numpy.cumsum(3.5938 * numpy.power(1.035012, range(200))) + 0.1
    X, Y = numpy.meshgrid(dx, numpy.linspace(0, 200, num=96))
    times = numpy.cumsum(numpy.power(1.421245, range(24))) / (365)

    # Unnormalize Data
    Input_Batch, Output_True_Batch = DataUnnormalization(Input_Batch.cpu(), Output_True_Batch.detach().cpu(), Params)
    Output_Pred_Batch = DataUnnormalization(Inputs=None, Outputs=Output_Pred_Batch.detach().cpu(), Params=Params)

    Output_True_Batch = Output_True_Batch.cpu().numpy()
    Output_Pred_Batch = Output_Pred_Batch.cpu().numpy()
    Input_Batch       = Input_Batch.cpu().numpy()

    # Build argument list for each dataset
    Args = [
        (Output_True_Batch[i], Output_Pred_Batch[i], Input_Batch[i],
         Dataset_ID_Batch[i], Result_Path, X, Y, times, Problem_Type)
        for i in range(len(Dataset_ID_Batch))
    ]

    # Loop over Dataset in parallel
    with multiprocessing.Pool() as Pool:
        Pool.starmap(Visualize_Single_Dataset, Args)

def PlotTrainingMetrics(Epoch_Tensor, MSE_Tensor, RMSE_Tensor, R2_Tensor, Result_Path, Model_Name):

    # Move Tensors to CPU and Convert to Numpy
    Epoch_Tensor = Epoch_Tensor.cpu().numpy()
    MSE_Tensor = MSE_Tensor.cpu().numpy()
    RMSE_Tensor = RMSE_Tensor.cpu().numpy()
    R2_Tensor = R2_Tensor.cpu().numpy()

    # Create Subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    # ---- LOSS ----
    axs[0].plot(Epoch_Tensor, MSE_Tensor[:, 0], color='r', linestyle='-')
    axs[0].plot(Epoch_Tensor, MSE_Tensor[:, 1], color='b', linestyle='--')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel(f'MSE')
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', axis='both')
    axs[0].set_xlim(0,Epoch_Tensor[-1])

    # ---- RMSE ----
    axs[1].plot(Epoch_Tensor, RMSE_Tensor[:, 0], color='r', linestyle='-')
    axs[1].plot(Epoch_Tensor, RMSE_Tensor[:, 1], color='b', linestyle='--')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('RMSE')
    axs[1].set_yscale('log')
    axs[1].grid(True, which='both', axis='both')
    axs[1].set_xlim(0,Epoch_Tensor[-1])

    # ---- R^2 ----
    axs[2].plot(Epoch_Tensor, R2_Tensor[:, 0], color='r', linestyle='-')
    axs[2].plot(Epoch_Tensor, R2_Tensor[:, 1], color='b', linestyle='--')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('R^2')
    axs[2].grid(True, which='both', axis='both')
    axs[2].set_xlim(0,Epoch_Tensor[-1])

    # Set Legend
    fig.legend(labels=['Training', 'Validation'], loc='lower center', ncol=3, frameon=True)
    
    # Save Plot 
    plt.savefig(os.path.join(Result_Path, f'{Model_Name}_Training_Metrics.png'), dpi = 600)
    plt.close()

    return




