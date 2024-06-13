import numpy as np
import torch

def calculate_range_and_divergence(variable, name):
    var_min = np.min(variable)
    var_max = np.max(variable)
    var_range = var_max - var_min
    var_std = np.std(variable)

    print(f'{name}')
    print('Type: ', type(variable), '  Shape: ', variable.shape)
    print(f'{name} Range: min = {var_min}, max = {var_max}, range = {var_range}')
    print(f'{name} Divergence (Standard Deviation): {var_std}')
    print('')

# Check for NaN values in the predicted variables
def check_for_nan_and_inf(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {tensor_name}")
    if torch.isinf(tensor).any():
        print(f"Inf. values found in {tensor_name}")
    else:
      print(tensor_name+', no inf, no Nan')

def check_tensor_stats(tensor, name):
    if tensor.numel() == 0:
        print(f"{name} is empty")
    else:
        print(f"{name}: mean={tensor.mean().item()}, std={tensor.std().item()}, min={tensor.min().item()}, max={tensor.max().item()}")

'''
input_min = [np.min(coords[:, 0]), np.min(coords[:, 1]), 9/80, np.min(Re), np.min(theta)]
input_max = [np.max(coords[:, 0]), np.max(coords[:, 1]), 900/80, np.max(Re), np.max(theta)]
'''

