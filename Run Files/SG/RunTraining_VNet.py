"""
====================================================================
File: RunTraining.py
Description:
    Main execution script for training and evaluating the NN specified
    by the Config_XXX.py for the Wen et al. (2022) Pressure Equation 
    dataset.

    This script loads configuration settings from Config_VNet.py,
    initializes the model, dataset, optimizer, and scheduler,
    and manages the full training and evaluation pipeline.
Author:  Giovanni Zingaro
Created: 2025-02-15
====================================================================
"""

import sys
import argparse
import time
from pathlib import Path

## Set Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    curDir = str(PROJECT_ROOT)

## Import Libraries
from Functions.Import import *
from Functions.Training.Main import Main

## Import Configuration
from Config.SG.Config_VNet import (
    Model_Config,
    Dataset_Config,
    Optimizer_Config,
    Scheduler_Config,
    Toggle_Config
)

## Parse Command Line Arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the NN for the Wen et al. (2022) Pressure Equation dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=Model_Config["Model_Name"],
        help=f'Name of the model (default: "{Model_Config["Model_Name"]}")'
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["dP", "SG"],
        default=Model_Config["Problem_Type"],
        help=f'Problem type: "dP" or "SG" (default: "{Model_Config["Problem_Type"]}")'
    )
    parser.add_argument(
        "--lambda_r",
        type=float,
        default=Model_Config["Lambda_r"],
        help=f'Scalar weight for first derivative radial loss term (default: {Model_Config["Lambda_r"]})'
    )
    parser.add_argument(
        "--lambda_z",
        type=float,
        default=Model_Config["Lambda_z"],
        help=f'Scalar weight for first derivative vertical loss term (default: {Model_Config["Lambda_z"]})'
    )
    return parser.parse_args()

## Device Setup
if torch.backends.mps.is_available():
    Device = torch.device("mps")
elif torch.cuda.is_available():
    Device = torch.device("cuda")
else:
    Device = torch.device("cpu")
print(f"\nDEVICE: {Device}")

## Run Training
if __name__ == "__main__":

    ## Override Model_Config with CLI args
    args = parse_args()
    Model_Config["Model_Name"]    = args.model_name
    Model_Config["Lambda_r"]      = args.lambda_r
    Model_Config["Lambda_z"]      = args.lambda_z

    ## Record Start Time
    start_time     = time.time()

    print(f"\n===== STARTING TRAINING =====")
    print(f"  Model Name   : {Model_Config['Model_Name']}")
    print(f"  Lambda_r     : {Model_Config['Lambda_r']}")
    print(f"  Lambda_z     : {Model_Config['Lambda_z']}")

    Main(curDir, Model_Config, Dataset_Config, Optimizer_Config, Scheduler_Config, Toggle_Config, Device)

    ## Record End Time
    end_time     = time.time()
    elapsed_hrs  = (end_time - start_time) / 3600

    print(f"\n===== TRAINING COMPLETE =====")
    print(f"  Elapsed Time : {elapsed_hrs:.4f} hrs")
    print(f"{'=' * 30}\n")