"""
====================================================================
File: RunPostProcessing.py
Description:
    This post processes each model used in this comparative study.
Author:  Giovanni Zingaro
Created: 2026-04-24
====================================================================
"""

# Import Libraries 
import sys
import argparse
import time
from pathlib import Path
import os

## Set Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    curDir = str(PROJECT_ROOT)

## Import Libraries
from Functions.Import import *
from Functions.PostProcessing.Main import Main

## Device Setup
if torch.backends.mps.is_available():
    Device = torch.device("mps")
elif torch.cuda.is_available():
    Device = torch.device("cuda")
    GPU_Count = torch.cuda.device_count()
else:
    Device = torch.device("cpu")
print(f"\nDEVICE: {Device}")

## Inputs to Post Processing 
Problem_Type = 'SG'                                           # Problem Type -> Options ["dP", "SG"]
Model_Types = ['FNO']
Dataset_Config = {'Batch_Data': False}                        

## Run Post Processing
if __name__ == "__main__":
    
    for Model_Type in Model_Types:

        ModelsDir = os.path.join(curDir, 'Saved Models', Problem_Type, Model_Type)
        Model_Names = [os.path.splitext(f)[0] for f in os.listdir(ModelsDir) if os.path.isfile(os.path.join(ModelsDir, f)) and not f.startswith(".")]

        # Loop over Models
        for Model_Name in Model_Names:

            ## Record Start Time
            start_time     = time.time()

            print(f"\n===== STARTING POST PROCESSING =====")
            print(f"  Model Name   : {Model_Name}")

            Main(curDir, Model_Name, Problem_Type, Model_Type, Dataset_Config, Device)

            # Record End Time
            end_time = time.time()

            print(f"\n===== COMPLETE =====")
            print(f"Time Elapsed: {(end_time-start_time):.2f} seconds")