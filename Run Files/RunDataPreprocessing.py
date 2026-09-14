"""
=====================================================================
File: RunDataProcessing.py
Description:
 Preprocesses data used later to train the surrogate model. Here, 
 the preprocess occurs on a per-property level. The standard preproc-
 essing procedure supports both logarithmic normalization followed
 by Z-score. 
Author:  Giovanni Zingaro
Created: 2025-11-28
=====================================================================
"""

## Import Libraries
import sys
from pathlib import Path

## Set Current Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    curDir = str(PROJECT_ROOT)

## Import Main File
from Functions.Preprocessing.Main import Main
from Functions.Import import *

## Set Config
## Inputs by Index -> [k_r, k_z, φ, b_perf, Q_inj, T_int, P_int, Swi, λ, r, z, t]

# Normalization Type: Supported Options -> 'Log', 'Z-Score' -> 'Log' refers to Log Normalziation before Z-Score Scaling
Norm_Type_Inputs = {'k_r': 'Z-Score',
                    'k_z': 'Z-Score',
                    'phi': 'Z-Score',
                    'b_perf': 'Z-Score',
                    'Q_inj': 'Z-Score',
                    'T_int': 'Z-Score',
                    'P_int': 'Z-Score',
                    'Swi': 'Z-Score',
                    'Lambda': 'Z-Score',
                    'r': 'Z-Score',
                    'z': 'Z-Score',
                    't': 'Z-Score'}
Norm_Type_Outputs = {'dP': 'Z-Score'}                         
Data_Split        = [0.75, 0.125, 0.125]                                # Training, Validation and Testing Ratios
Norm_Dataset_Name = 'dP_Dataset'                                        # Normalized Dataset Name 
Dataset_Name      = ['dP_val_a.pt', 'dP_val_u.pt']                      # Dataset Name      
Batch_Size        = 500                                                  # Batch Size
Save_Params       = True                                                # Toggle to Save Normalization Parameters
Terminal_Toggle   = True                                                # Print Messages to Terminal

if __name__ == "__main__":
    Main(curDir, Norm_Type_Inputs, Norm_Type_Outputs, Data_Split, Norm_Dataset_Name, Dataset_Name, Batch_Size, Save_Params, Terminal_Toggle)



