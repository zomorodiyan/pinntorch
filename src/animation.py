import torch
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

model_files = [f'../models/c35_{i*10-9}_{i*10}.pth' for i in range(1, 11)]

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
        self.elements_per_snapshot = 5*13001
        self.elements_per_simulation = 13001

    def get_plotting_data(self, snapshot, simulation):
        start_idx = (snapshot - 1) * self.elements_per_snapshot +\
          (simulation - 1) * self.elements_per_simulation
        end_idx = start_idx + self.elements_per_simulation
        return self.data[start_idx:end_idx]

def load_model(model_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN().to(device)
    model.load_state_dict(torch.load(model_file))
    return model

def plot_fields(x, y, u, v, p, k, omega, c, snapshot,
              simulation,  name = 'new_fig', save_dir="../animations/figures", U_star = None):
    with torch.no_grad():
        L_star = 80.0

        # Convert predictions to numpy arrays for plotting
        u = u.cpu().numpy()
        v = v.cpu().numpy()
        p = p.cpu().numpy()
        k = k.cpu().numpy()
        omega = omega.cpu().numpy()
        c = c.cpu().numpy()
        x = x.cpu()
        y = y.cpu()

        radius = 40.0 / L_star
        # dimensionalize the predictions if U_star is provided
        if U_star is not None:
            x = x * L_star
            y = y * L_star
            u = u * U_star
            v = v * U_star
            p = p * U_star**2
            k = k * U_star**2
            omega = omega * U_star / L_star
            radius = 40.0

        # Triangulation for plotting
        triang = tri.Triangulation(x.squeeze(), y.squeeze())
        x_tri = x[triang.triangles].mean(axis=1)
        y_tri = y[triang.triangles].mean(axis=1)

        # Mask the triangles inside the circle
        center = (0.0, 0.0)
        dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
        mask = dist_from_center < radius
        mask = mask.squeeze()
        mask = mask.cpu().numpy().astype(bool)
        triang.set_mask(mask)


        # Plotting details stored in a dictionary
        plot_details = [
            {'data': u, 'levels': np.linspace(-6, 20, 100), 'ticks': np.linspace(-6, 18, 7),\
             'title': f'Simulation $u$ at time {snapshot}$s$ [$m/s$]', 'scientific': False},
            {'data': v, 'levels': np.linspace(-9, 9, 100), 'ticks': np.linspace(-9, 9, 7),\
             'title': f'Simulation $v$ at time {snapshot}$s$ [$m/s$]', 'scientific': False},
            {'data': p, 'levels': np.linspace(-160, 60, 100), 'ticks': np.linspace(-160, 60, 7),\
             'title': f'Simulation $p$ at time {snapshot}$s$ [pa]', 'scientific': False},
            {'data': k, 'levels': np.linspace(0, 5, 100), 'ticks': np.linspace(0, 5, 7),\
             'title': f'Simulation $k$ at time {snapshot}$s$ [$m^2/s^2$]', 'scientific': False},
            {'data': omega, 'levels': np.linspace(0, 3.2, 100), 'ticks': np.linspace(0, 3.2, 7),\
             'title': f'Simulation $\omega$ at time {snapshot}$s$ [$s⁻¹$]', 'scientific': False},
            {'data': c, 'levels': np.linspace(0, 6e-5, 100), 'ticks': np.linspace(0, 6e-5, 7),\
             'title': f'Simulation $c$ at time {snapshot}$s$ []', 'scientific': True} ]

        # Plotting
        kk = 1
        fig1 = plt.figure(figsize=(kk*18, kk*12))

        for i, detail in enumerate(plot_details):
            plt.subplot(3, 2, i+1)
            contour = plt.tricontourf(triang, detail['data'].squeeze(), cmap='jet', levels=detail['levels'])
            cbar = plt.colorbar(contour)
            cbar.set_ticks(detail['ticks'])
            plt.title(detail['title'])
            plt.tight_layout()

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=400)

        return fig1

def create_animation(models, dataset, start_time, end_time, simulation,\
  save_dir="../animations", filename="animation.gif"):
    frames = []
    os.makedirs(save_dir, exist_ok=True)

    print(' ---- creating animation ---- ')
    for time_step in range(start_time, end_time + 1):
        print(f'time step {time_step}')
        plot_data = dataset.get_plotting_data(time_step, simulation).to(device)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u_, v_, p_, k_, omega_, c_ = plot_outputs
#       with torch.no_grad():
#         u, v, p, k, omega, c = models[(time_step-1)//10](torch.cat([x, y, t, Re, theta], dim=1))
        # Plot the fields

#       fig = plot_fields(x, y, torch.abs(u-u_.squeeze()), torch.abs(v-v_.squeeze()), torch.abs(p-p_.squeeze()), torch.abs(k-k_.squeeze()), torch.abs(omega-omega_.squeeze()), torch.abs(c-c_.squeeze()), time_step, simulation, name=f"fig_{time_step}", save_dir=save_dir, U_star = 9.0)
#       fig = plot_fields(x, y, u, v, p, k, omega_, c_, time_step, simulation, name=f"fig_j6_{time_step}", save_dir=save_dir, U_star = 9.0)
        fig = plot_fields(x, y, u_.squeeze(), v_.squeeze(), p_.squeeze(), k_.squeeze(), omega_.squeeze(), c_.squeeze(), time_step, simulation, name=f"fig_j6_{time_step}", save_dir=save_dir, U_star = 9.0)
        plt.close(fig)  # Close the figure to save memory

        # Save the figure as an image
        image_path = os.path.join(save_dir, f"fig_{time_step}.png")
        frames.append(image_path)

    # Create the animation
#   images = [Image.open(frame) for frame in frames]
#   images[0].save(os.path.join(save_dir, filename), save_all=True, append_images=images[1:], duration=200, loop=0, optimize=False, quality=100)

models = [load_model(model_file) for model_file in model_files]
data_array = np.load("../data/preprocessed_clipped.npy")
dataset = CustomDataset(data_array)
create_animation(models, dataset, start_time=1, end_time=100, simulation=1)
#sim1: north_east_leak
