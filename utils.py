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
    is_nan = torch.isnan(tensor).any()
    is_inf = torch.isinf(tensor).any()
    if is_nan and is_inf:
      print(tensor_name+'  Nan & Inf found')
    elif is_nan:
        print(tensor_name+'  Nan found')
    elif is_inf:
        print(tensor_name+'  Inf found')
    else:
        print(tensor_name+'----------------')

def check_tensor_stats(tensor, name):
    if tensor.numel() == 0:
        print(f"{name} is empty")
    else:
        print(f"{name}: mean={tensor.mean().item()}, std={tensor.std().item()}, min={tensor.min().item()}, max={tensor.max().item()}")

'''
input_min = [np.min(coords[:, 0]), np.min(coords[:, 1]), 9/80, np.min(Re), np.min(theta)]
input_max = [np.max(coords[:, 0]), np.max(coords[:, 1]), 900/80, np.max(Re), np.max(theta)]
'''

