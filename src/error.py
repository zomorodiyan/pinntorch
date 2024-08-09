import torch
import time
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os
#import imageio
#import imageio.v2 as imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_files = ['../models/c35_single_100s_3.pth']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        N1 = 128  # Number of neurons
        self.normalization = NormalizationLayer()
        self.fourier_embedding = FourierEmbedding(input_dims=5, embed_dims=256,
                                                  scale=1.0)
        self.fc1 = nn.Linear(256, N1)
        self.fc2 = nn.Linear(N1, N1)
        self.fc3 = nn.Linear(N1, N1)
        self.fc4 = nn.Linear(N1, N1)
#       self.fc5 = nn.Linear(N1, N1)
        self.fc_out = nn.Linear(N1, 6)  # Combine outputs into a single layer
        self.denormalization = DenormalizationLayer()
        self.initialize_weights()

    def forward(self, x):
        x = self.normalization(x)
        x = self.fourier_embedding(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
#       x = F.gelu(self.fc5(x))
        outputs = self.fc_out(x)
        outputs = self.denormalization(outputs)
        u, v, p, k, omega, c = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4], outputs[:, 5]
        return [u, v, p, k, omega, c]

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()
        self.input_min = torch.tensor([-2.5, -2.5, 1, 30e6, 0], device=device)
        self.input_max = torch.tensor([7.5, 2.5, 100, 70e6, 2 * np.pi], device=device)
    def forward(self, x):
        if self.input_min.device != x.device:
            self.input_min = self.input_min.to(x.device)
        if self.input_max.device != x.device:
            self.input_max = self.input_max.to(x.device)
        return (x - self.input_min) / (self.input_max - self.input_min)

class DenormalizationLayer(nn.Module):
    def __init__(self):
        super(DenormalizationLayer, self).__init__()
        self.output_min = torch.tensor([-1, -1, -2, 1e-12, 1e-12, 1e-12], device=device)
        self.output_max = torch.tensor([2, 1, 1, 0.1, 25, 5e-5], device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.output_min.device != x.device:
            self.output_min = self.output_min.to(x.device)
        if self.output_max.device != x.device:
            self.output_max = self.output_max.to(x.device)
        # Apply ReLU to the last three outputs: k, omega, c (non-negative outputs)
        x[:, 3:] = self.relu(x[:, 3:])
        return x * (self.output_max - self.output_min) + self.output_min

class FourierEmbedding(nn.Module):
    def __init__(self, input_dims, embed_dims, scale=1.0):
        super(FourierEmbedding, self).__init__()
        self.input_dims = input_dims
        self.embed_dims = embed_dims
        self.scale = scale
        self.B = nn.Parameter(self.scale * torch.randn(input_dims, embed_dims // 2), requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def prepare_inputs_outputs(batch_data):
    inputs = [batch_data[:, i:i+1] for i in range(5)]
    outputs = [batch_data[:, i+5:i+6] for i in range(6)]
    return inputs, outputs

class CustomDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32).to(device)
        self.elements_per_snapshot = 13001
        self.elements_per_simulation = 13001

    def get_plotting_data(self, snapshot, simulation):
        start_idx = (snapshot - 1) * self.elements_per_snapshot +\
          (simulation - 1) * self.elements_per_simulation
        end_idx = start_idx + self.elements_per_snapshot
        return self.data[start_idx:end_idx]

def load_model(model_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN().to(device)
    model.load_state_dict(torch.load(model_file))
    return model

def create_animation(models, dataset, start_time, end_time, simulation,\
  save_dir="../animations", filename="animation.gif"):
    frames = []
    os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store MAE for each variable
    mae_u_list = []
    mae_v_list = []
    mae_p_list = []
    mae_k_list = []
    mae_omega_list = []
    mae_c_list = []

# Initialize lists to store MSE for each variable
    mse_u_list = []
    mse_v_list = []
    mse_p_list = []
    mse_k_list = []
    mse_omega_list = []
    mse_c_list = []

    print(' ---- creating animation ---- ')
    for time_step in range(start_time, end_time + 1):
        print(f'time step {time_step}')
        plot_data = dataset.get_plotting_data(time_step, simulation).to(device)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        print('x.shape',x.shape)
        u_, v_, p_, k_, omega_, c_ = plot_outputs
        with torch.no_grad():
          u, v, p, k, omega, c = models[0](torch.cat([x, y, t, Re, theta], dim=1))

        # Calculate MAE for each variable
        mae_u = torch.mean(torch.abs(u - u_)).item()
        mae_v = torch.mean(torch.abs(v - v_)).item()
        mae_p = torch.mean(torch.abs(p - p_)).item()
        mae_k = torch.mean(torch.abs(k - k_)).item()
        mae_omega = torch.mean(torch.abs(omega - omega_)).item()
        mae_c = torch.mean(torch.abs(c - c_)).item()

        mse_u = torch.mean((u - u_) ** 2).item()
        mse_v = torch.mean((v - v_) ** 2).item()
        mse_p = torch.mean((p - p_) ** 2).item()
        mse_k = torch.mean((k - k_) ** 2).item()
        mse_omega = torch.mean((omega - omega_) ** 2).item()
        mse_c = torch.mean((c - c_) ** 2).item()

        # Append MAE to respective lists
        mae_u_list.append(mae_u)
        mae_v_list.append(mae_v)
        mae_p_list.append(mae_p)
        mae_k_list.append(mae_k)
        mae_omega_list.append(mae_omega)
        mae_c_list.append(mae_c)

        # Append MSE to respective lists
        mse_u_list.append(mse_u)
        mse_v_list.append(mse_v)
        mse_p_list.append(mse_p)
        mse_k_list.append(mse_k)
        mse_omega_list.append(mse_omega)
        mse_c_list.append(mse_c)

    fig, axs = plt.subplots(6, 1, figsize=(10, 15))
    time_steps = list(range(start_time, end_time+1))

# Define the split point for solid and dashed lines
    split_point = 68

# Plotting function with conditional linestyle
    def plot_with_styles(ax, time_steps, data_list, label, color):
        ax.plot(time_steps[:split_point], data_list[:split_point], label=label, color=color, linestyle='-')
        ax.plot(time_steps[split_point:], data_list[split_point:], label=f'{label} (no data)', color=color, linestyle='--')
#       ax.set_ylabel(f'MAE ({label})')
        ax.set_ylabel(f'MSE ({label})')
        ax.legend()

#   plot_with_styles(axs[0], time_steps, np.array(mae_u_list), 'u', 'b')
#   plot_with_styles(axs[1], time_steps, np.array(mae_v_list), 'v', 'g')
#   plot_with_styles(axs[2], time_steps, np.array(mae_p_list), 'p', 'r')
#   plot_with_styles(axs[3], time_steps, np.array(mae_k_list), 'k', 'c')
#   plot_with_styles(axs[4], time_steps, np.array(mae_omega_list), '$\omega$', 'm')
#   plot_with_styles(axs[5], time_steps, np.array(mae_c_list), 'c', 'y')

    plot_with_styles(axs[0], time_steps, np.array(mse_u_list), 'u', 'b')
    plot_with_styles(axs[1], time_steps, np.array(mse_v_list), 'v', 'g')
    plot_with_styles(axs[2], time_steps, np.array(mse_p_list), 'p', 'r')
    plot_with_styles(axs[3], time_steps, np.array(mse_k_list), 'k', 'c')
    plot_with_styles(axs[4], time_steps, np.array(mse_omega_list), '$\omega$', 'm')
    plot_with_styles(axs[5], time_steps, np.array(mse_c_list), 'c', 'y')

    axs[5].set_xlabel('Time Step')

    plt.tight_layout()
#   plt.title('Mean Absolute Error of variables over Time')
    plt.title('Mean Squared Error of variables over Time')
    plt.savefig('mse.png', dpi=400)

models = [load_model(model_file) for model_file in model_files]
data_array = np.load("../data/single.npy")
dataset = CustomDataset(data_array)
create_animation(models, dataset, start_time=1, end_time=100, simulation=1)
#sim1: north_east_leak
