import numpy as np
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
