# Functions/Visualization/VisualizeData.py

# Import Libraries 
from Functions.Import import *

# Pressure Normalization
def dnorm_dP(dP):
    dP = dP * 18.772821433027488
    dP = dP + 4.172939172019009 
    return dP

# Plot Property Function 
def PlotProperty(X, Y, Thickness, Property, Property_Name, fig, axis, x_lim, tick_increment):
        
    pcm = axis.pcolor(X[:Thickness, :], Y[:Thickness, :], numpy.flipud(Property), shading='auto', cmap='jet')
    fig.colorbar(pcm, ax=axis, label=Property_Name)
    axis.set_xlim([0, x_lim])
    axis.set_xticks(numpy.arange(0, x_lim + 1, tick_increment))
    axis.set_xlabel('r [m]')
    axis.set_xlabel('z [m]')

    return 

# Visualize Input Function
def VisualizeInputs(Inputs, curDir, Dataset_ID, Visualize_Toggle, Rewrite_Toggle, Problem_Type):

    # Extract Mask and Thickness
    Mask = Inputs[:, :, 0, 0] != 0  # Mask 
    Thickness = sum(Mask[:, 0])     # Thickness of Reservoir
    np = Inputs.shape[-1]           # Number of Properties

    # Unnormalization Functions 
    dnorm_inj = lambda a : (a * (3e6 - 3e5) + 3e5) / (1e6 /365*1000/1.862)
    dnorm_temp = lambda a : a * (180 - 30) + 30
    dnorm_P = lambda a : a * (300 - 100) + 100
    dnorm_lam = lambda a : a * 0.4 + 0.3
    dnorm_Swi = lambda a : a * 0.2 + 0.1

    # Time Vector and Rewrite (Currently in incremental form)
    nt = Inputs.shape[2]
    times = torch.from_numpy(numpy.cumsum(numpy.power(1.421245, range(nt)))) 

    # Define Spatial Coordinates
    dx = numpy.cumsum(3.5938*numpy.power(1.035012, range(200))) + 0.1
    X, Y = numpy.meshgrid(dx, numpy.linspace(0,200, num = 96))

    # Rewrite Inputs
    if Rewrite_Toggle is True:

        # Extract Input Parameters
        poro_map = Inputs[:,:,0,2][Mask].reshape((Thickness, -1))
        kr_map = numpy.exp(Inputs[:,:,0,0][Mask].reshape((Thickness, -1))*15)
        kz_map = numpy.exp(Inputs[:,:,0,1][Mask].reshape((Thickness, -1))*15)
        perf_map = Inputs[:,:,0,3][Mask].reshape((Thickness, -1)).float()
        inj_rate = dnorm_inj(Inputs[:,:,:,4])
        temperature = dnorm_temp(Inputs[:,:,:,5])
        pressure = dnorm_P(Inputs[:,:,:,6])
        Swi = dnorm_Swi(Inputs[:,:,:,7])
        lam = dnorm_lam(Inputs[:,:,:,8])

        Inputs[:, :, :, 2][Mask] = (poro_map.ravel()[:, None]).expand(-1, 24)
        Inputs[:, :, :, 0][Mask] = (kr_map.ravel()[:, None]).expand(-1, 24)
        Inputs[:, :, :, 1][Mask] = (kz_map.ravel()[:, None]).expand(-1, 24)
        Inputs[:, :, :, 3][Mask] = (perf_map.ravel()[:, None]).expand(-1, 24)
        Inputs[:, :, :, 4]  = inj_rate
        Inputs[:, :, :, 5]  = temperature
        Inputs[:, :, :, 6]  = pressure
        Inputs[:, :, :, 7]  = Swi
        Inputs[:, :, :, 8]  = lam
        Inputs[:, :, :, 9]  = torch.tensor(X).unsqueeze(-1).expand(96, 200, 24)  
        Inputs[:, :, :, 10]  = torch.tensor(Y).unsqueeze(-1).expand(96, 200, 24)  
        Inputs[:, :, :, 11] = times.view(1, 1, nt)

        # Apply Mask 
        Inputs[:,:,:,:][~Mask] = -99

    # Plot Inputs
    if Visualize_Toggle is True:

        # Define Subplot
        fig, axis = plt.subplots(4, 3, figsize=(12, 5))

        # Plot Properties
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,0][Mask].reshape((Thickness, -1)), '$k_r$ [mD]', fig, axis[0,0], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,1][Mask].reshape((Thickness, -1)), '$k_z$ [mD]', fig, axis[0,1], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,2][Mask].reshape((Thickness, -1)), r'$\phi$ [-]', fig, axis[0,2], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,3][Mask].reshape((Thickness, -1)), '$b_{Perf}$ [m]', fig, axis[1,0], x_lim = 50, tick_increment=10)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,4][Mask].reshape((Thickness, -1)), '$Q_{Inj}$ [Mt/yr]', fig, axis[1,1], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,5][Mask].reshape((Thickness, -1)), '$T$ [°C]', fig, axis[1,2], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,6][Mask].reshape((Thickness, -1)), '$P_{Init}$ [bar]', fig, axis[2,0], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,7][Mask].reshape((Thickness, -1)), '$S_{wi}$ [-]', fig, axis[2,1], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,8][Mask].reshape((Thickness, -1)), r'$\lambda$ [-]', fig, axis[2,2], x_lim = 3500, tick_increment=1000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,9][Mask].reshape((Thickness, -1)), '$r$ [m]', fig, axis[3,0], x_lim = 100000, tick_increment=25000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,10][Mask].reshape((Thickness, -1)), '$z$ [m]', fig, axis[3,1], x_lim = 100000, tick_increment=25000)
        PlotProperty(X, Y, Thickness, Inputs[:,:,0,11][Mask].reshape((Thickness, -1)), '$t$ [day]', fig, axis[3,2], x_lim = 100000, tick_increment=25000)
        plt.tight_layout()
    
        # Save Figure
        Save_Path = os.path.join(curDir, 'Data Plots', str(Problem_Type), f'Dataset {Dataset_ID}')
        os.makedirs(Save_Path, exist_ok=True)
        plt.savefig(os.path.join(Save_Path, f"Properties_Plot.png"), dpi=600, bbox_inches='tight')
        plt.close(fig)  

    return Inputs

# Plot Output Function 
def PlotOutput(X, Y, Thickness, Output, fig_title, colour_bar_label):
    fig = plt.figure(figsize=(15,6))
    Z = numpy.ma.masked_invalid(numpy.flipud(Output))
    pcm = plt.pcolor(X[:Thickness, :], Y[:Thickness, :], Z, shading='auto', vmin=0)
    plt.jet()
    plt.colorbar(pcm, fraction=0.02, label=colour_bar_label)
    plt.title(fig_title)
    plt.xlim([0, 3500])
    plt.xticks(numpy.arange(0, 3500 + 1, 500))
    fig.tight_layout()
    return fig


# Visualize Input Function
def VisualizeOutputs(Inputs, Outputs, curDir, Dataset_ID, Visualize_Toggle, Problem_Type):

    # Mask and Thickness
    Mask = Inputs[:, :, 0, 0] != -99  # Mask 
    Thickness = sum(Mask[:, 0])         # Thickness of Reservoir

    # Spatial Grid
    dx = numpy.cumsum(3.5938*numpy.power(1.035012, range(200))) + 0.1
    X, Y = numpy.meshgrid(dx, numpy.linspace(0,200,num=96))

    Save_Path = os.path.join(curDir, 'Data Plots', str(Problem_Type), f'Dataset {Dataset_ID}')
    os.makedirs(Save_Path, exist_ok=True)

    times = numpy.cumsum(numpy.power(1.421245, range(24)))
    t = times[-1]

    if t / 365 < 1:
        t_label = f"{t:0.1f} days"
        t_file  = f"{t:0.1f}_days.png"
    else:
        t_yrs = t / 365
        t_label = f"{t_yrs:0.2f} yrs"
        t_file  = f"{t_yrs:0.2f}_yrs.png"

    # Plot Output Results
    if Visualize_Toggle is True:

        # Pressure Build-up
        if Problem_Type == "dP":
            #Outputs = dnorm_dP(Outputs)
            fig = PlotOutput(X, Y, Thickness, Outputs[:,:,-1][Mask].reshape((Thickness, -1)), f'dP [bar], t = {t_label}', 'dP [bar]')
            plt.savefig(os.path.join(Save_Path, f'dP_Plot_Dataset_{t_file}'),
                    dpi=600, bbox_inches='tight')

        # Gas Saturation
        elif Problem_Type == "SG":
            fig = PlotOutput(X, Y, Thickness, Outputs[:,:,-1][Mask].reshape((Thickness, -1)), f'SG [-], t = {t_label}', 'SG [-]')
            plt.savefig(os.path.join(Save_Path, f'SG_Plot_Dataset_{t_file}'),
                    dpi=600, bbox_inches='tight')

        plt.close(fig)

    # Restate Mask
    Outputs[~Mask] = -99

    return Outputs

