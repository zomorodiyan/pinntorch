import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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
        check_tensor_stats(tensor, tensor_name)

def check_tensor_stats(tensor, name):
    if tensor.numel() == 0:
        print(f"{name} is empty")
    else:
        print(f"{name: <10} mean={tensor.mean().item(): .3e}, std={tensor.std().item(): .3e}, min={tensor.min().item(): .3e}, max={tensor.max().item(): .3e}")


def plot_fields(time, model, x, y, t, Re, theta, name = 'new_fig', U_star = None, save_dir="figures"):
    with torch.no_grad():
        L_star = 80.0

        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(\
          torch.cat([x_pred, y_pred, t_pred, Re_pred, theta_pred], dim=1))

        # Convert predictions to numpy arrays for plotting
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        p_pred = p_pred.cpu().numpy()
        k_pred = k_pred.cpu().numpy()
        omega_pred = omega_pred.cpu().numpy()
        c_pred = c_pred.cpu().numpy()
        x_pred = x_pred.cpu()
        y_pred = y_pred.cpu()

        # dimensionalize the predictions if U_star is provided
        if U_star is not None:
            u_pred = u_pred * U_star
            v_pred = v_pred * U_star
            p_pred = p_pred * U_star**2
            k_pred = k_pred * U_star**2
            omega_pred = omega_pred * U_star / L_star

        # Triangulation for plotting
        triang = tri.Triangulation(x_pred.squeeze(), y_pred.squeeze())

        # Mask the triangles inside the circle
        center = (0.0, 0.0)
        radius = 40.0 / L_star

        x_tri = x_pred[triang.triangles].mean(axis=1)
        y_tri = y_pred[triang.triangles].mean(axis=1)

        dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
        mask = dist_from_center < radius
        mask = mask.squeeze()
        mask = mask.cpu().numpy().astype(bool)
        triang.set_mask(mask)

        # Plotting
        fig1 = plt.figure(figsize=(18, 12))

        plt.subplot(3, 2, 1)
        plt.tricontourf(triang, u_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $u$ at time {snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.tricontourf(triang, v_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $v$ at time {snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.tricontourf(triang, p_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $p$ at time {snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.tricontourf(triang, k_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $k$ at time {snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, omega_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $\omega$ at time {snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.tricontourf(triang, c_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $c$ at time {snapshot}s')
        plt.tight_layout()

        # Save the figure
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{name}_fields_t_{snapshot}_sim_{simulation}.png"))

        return fig1

def formatted_print(*args):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):
            formatted_args.append(f"{arg:.3g}")
        else:
            formatted_args.append(str(arg))
    print(*formatted_args)

