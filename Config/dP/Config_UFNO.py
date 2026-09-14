"""
====================================================================
File: Config_UFNO.py
Description:
    Configuration module defining model architecture, dataset,
    optimizer, scheduler, and training settings for the U-FNO
    applied to the Wen et al. (2022) Pressure Equation dataset.
Author:  Giovanni Zingaro
Created: 2025-02-15
====================================================================
"""

# ============================================================
# MODEL CONFIGURATION
# ============================================================
Model_Config = {
    "In_Channels": 12,                                          # Number of Inputs
    "Out_Channels": 1,                                          # Number of Outputs
    "Modes_1": 10,                                              # Number of Fourier Modes (x)
    "Modes_2": 10,                                              # Number of Fourier Modes (y)
    "Modes_3": 10,                                              # Number of Fourier Modes (t)
    "Width": 36,                                                # Channel Width
    "dh": 128,                                                  # Number of Neurons in Hidden Space
    "Model_Type": "U-FNO",                                        # Options: ['FNO', 'U-FNO', 'U-Net', 'CNN']
    "N_Epoch": 350,                                             # Number of Epochs
    "Model_Name": "U-FNO (dP Full Dataset)",                          # Model Name
    "Problem_Type": "dP",                          # Type of Model -> Options: ["dP", "SG"]
    "Lambda_r": 0.0,                                # Weight for Radial Derivative Loss Term   -> Total Loss = MSE(u) + Lambda_r * L_r + Lambda_z * L_z
    "Lambda_z": 0.0                                 # Weight for Vertical Derivative Loss Term  -> Total Loss = MSE(u) + Lambda_r * L_r + Lambda_z * L_z
}
# ============================================================
# DATASET CONFIGURATION
# ============================================================
Dataset_Config = {
    "Dataset_Name_Inputs": "dP_Raw_Inputs.pt",
    "Dataset_Name_Outputs": "dP_Raw_Outputs.pt",    
    "Batch_Size": 10,
    "Load_Data": True
}

# ============================================================
# OPTIMIZER CONFIGURATION
# ============================================================
Optimizer_Config = {    
    "Optimizer_Type": "Adam",               # Optimizer Type -> Options ['Adam']
    "Learning_Rate": 1e-3,                  # Learning Rate
    "Weight_Decay": 1e-4,                   # L2-Regularization Constant
    "Betas": (0.9, 0.999),                  # Beta (β1, β2)
    "Eps": 1e-8,                            # Epsilon 
}

# ============================================================
# SCHEDULER CONFIGURATION
# ============================================================
Scheduler_Config = {
    "T_0":    25,    
    "T_mult":  2,    
    "Eta_Min": 1e-6
}

# ============================================================
# TOGGLE CONFIGURATION
# ============================================================
Toggle_Config = {
    "Visual_Results": True,                 # Visualization Toggle
    "Plot_Rate": 50,                       # Number of Epochs for Plotting
    "Multi_GPU": False                       # Multi-GPU Training Toggle
}