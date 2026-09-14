"""
====================================================================
File: Functions/Imports.py
Description:
    Import Module Function for both Preprocessing and Training
====================================================================
"""

# Libraries
import os
import sys
from pathlib import Path
import gc

## Third Party Libraries
import numpy 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F   
from torch.utils.data import DataLoader, Dataset
import shutil
import copy
import multiprocessing as mp
from torchsummary import summary
import scipy
from scipy.io import savemat
import csv

# Plotting Libraries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

## Custom Functions
from Functions.Visualization.VisualizeData import VisualizeInputs, VisualizeOutputs
from Functions.Preprocessing.DataPreprocessing import DataNormalization, DataUnnormalization, CustomDataset
from Functions.Training.Train import Train
from Functions.Training.ComputeErrors import Compute_MSE, Compute_RMSE, Compute_R2, Compute_Regression_Metrics, Compute_Loss
from Functions.Training.PlotResults import PlotTrainingMetrics


# Public API
__all__ = [
    # Libraries
    "os", "sys", "Path", "gc",
    # Third Party
    "numpy", "tqdm", "DataLoader", "Dataset", "torch", "shutil", "F", "nn", "copy", "mp", "summary", "scipy", "csv", "savemat",
    # Plotting
    "matplotlib", "plt", "Normalize", "ScalarMappable", "Patch", "LogNorm", 
    # Custom Functions
    "VisualizeInputs", "VisualizeOutputs", "DataNormalization", "DataUnnormalization", "CustomDataset",
    "Train", 
    # Loss Function
    "Compute_MSE", "Compute_RMSE", "Compute_R2", "Compute_Regression_Metrics", "Compute_Loss",
    # Plot Results
    "PlotTrainingMetrics",
]