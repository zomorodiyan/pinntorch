import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from ml_collections import ConfigDict
from torch.profiler import profile, record_function, ProfilerActivity
import time
import random

print('run J23 ------------------------- 1 ---------------------------------')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2024)

# Global constants for SST k-omega model
a1 = 0.31
kappa = 0.41
alpha_0 = 1.0 / 9.0
alpha_star_infinity = 1.0
beta_i1 = 0.075
beta_i2 = 0.0828
beta_star_infinity = 0.09
sigma_omega1 = 2.0
R_beta = 8
sigma_omega2 = 1.168
R_k = 6
sigma_k1 = 1.176
R_omega = 2.95
sigma_k2 = 1.0
xi_star = 1.5
M_t0 = 0.25
f_beta = 1.0
f_star_beta = 1.0

# Physical constants
rho = 1.225  # kg/m^3
mu = 1.7894e-5  # kg/(m*s)
gamma = 1.4
R = 287  # m^2/(s^2*K)
T = 297.15  # K
L_star = 80.0 # [m] diameter
mu_rho_l = mu / (rho * L_star)

config = ConfigDict()
config.optim = optim_config = ConfigDict()
optim_config.optimizer = "Adam"
optim_config.beta1 = 0.9
optim_config.beta2 = 0.999
optim_config.eps = 1e-8
optim_config.learning_rate = 1e-3
optim_config.decay_rate = 0.9
optim_config.decay_steps = 500
optim_config.grad_accum_steps = 0
optim_config.clip_norm = 1000.0
optim_config.weight_decay = 1.0e-4

N_loss_print = 1
N_save_model = 5000
N_weight_update = 10
N_plot_fields = 1000
N_plot_tweight = 100
N_intervals = 10
N_log_metrics = 1

def save_model(model, path):
    torch.save(model.state_dict(), path)

def generate_boundary_conditions(interval, num_samples):
    Re_medium = 1.225 * 9.0 * 80.0 / 1.7894e-5
    U_in = 1.0 # N_star ≡ U_in so the non-dim version is 1.0
    k_in_value = 1.0#3 / 2 * (0.05 * U_in) ** 2
    omega_in_value = 25 # actual omega_in is higher but I have cliped omega > 25

    t_low, t_high = interval * (100 // N_intervals), (interval + 1) * (100 // N_intervals)
    theta_values = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
                                 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi], device=device)
    N_each_bc = num_samples
    boundary_conditions = []
    x_in = torch.full((N_each_bc, 1), -200.0 / L_star).to(device)
    y_in = torch.linspace(-200 / L_star, 200 / L_star, N_each_bc).view(-1, 1).to(device)
    t_in = torch.randint(t_low, t_high, x_in.size()).float().to(device)
    Re_in = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_in = theta_values[torch.randint(len(theta_values), x_in.size(), device=device)]
    inputs_in = (x_in, y_in, t_in, Re_in, theta_in)
    conditions_in = {
        'u': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), 1.0).to(device)},
        'v': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).to(device)},
        'k': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), k_in_value).to(device)},
        'omega': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), omega_in_value).to(device)}
    }

    x_sym = torch.linspace(-200 / L_star, 600 / L_star, N_each_bc).view(-1, 1).to(device)
    y_sym = (torch.where(torch.randint(0, 2, (N_each_bc, 1), device=device) == 0, -200.0, 200.0) / L_star).to(device)
    t_sym = torch.randint(t_low, t_high, x_sym.size()).float().to(device)
    Re_sym = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_sym = theta_values[torch.randint(len(theta_values), x_sym.size(), device=device)]
    inputs_sym = (x_sym, y_sym, t_sym, Re_sym, theta_sym)
    conditions_sym = {
        'u': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).to(device)},
        'v': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).to(device)},
        'p': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).to(device)},
        'k': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).to(device)},
        'omega': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).to(device)},
        'v_dir': {'type': 'Dirichlet', 'value': torch.zeros_like(x_sym).to(device)}
    }

    x_out = torch.full((N_each_bc, 1), 600.0 / L_star).to(device)
    y_out = torch.linspace(-200 / L_star, 200 / L_star, N_each_bc).view(-1, 1).to(device)
    t_out = torch.randint(t_low, t_high, x_out.size()).float().to(device)
    Re_out = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_out = theta_values[torch.randint(len(theta_values), x_out.size(), device=device)]
    inputs_out = (x_out, y_out, t_out, Re_out, theta_out)
    conditions_out = {
        'p': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).to(device)}
    }

    theta_rand = torch.linspace(0, 2 * np.pi, N_each_bc).to(device)
    x_wall = (40 / L_star * torch.cos(theta_rand)).view(-1, 1).to(device)
    y_wall = (40 / L_star * torch.sin(theta_rand)).view(-1, 1).to(device)
    t_wall = torch.randint(t_low, t_high, x_wall.size()).float().to(device)
    Re_wall = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_wall = theta_values[torch.randint(len(theta_values), x_wall.size(), device=device)]
    inputs_wall = (x_wall, y_wall, t_wall, Re_wall, theta_wall)
    conditions_wall = {
        'u': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).to(device)},
        'v': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).to(device)},
        'k': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).to(device)}
    }

    boundary_conditions.append((inputs_in, conditions_in))
    boundary_conditions.append((inputs_sym, conditions_sym))
    boundary_conditions.append((inputs_out, conditions_out))
    boundary_conditions.append((inputs_wall, conditions_wall))

    return boundary_conditions

class PINN(nn.Module):
    def __init__(self, input_min, input_max, output_min, output_max):
        super(PINN, self).__init__()
        N1 = 256  # Number of neurons
        self.normalization = NormalizationLayer(input_min, input_max)
        self.fourier_embedding = FourierEmbedding(input_dims=5, embed_dims=256, scale=1.0)
        self.fc1 = nn.Linear(256, N1)
        self.fc2 = nn.Linear(N1, N1)
        self.fc3 = nn.Linear(N1, N1)
        self.fc4 = nn.Linear(N1, N1)
        self.fc5 = nn.Linear(N1, N1)
        self.fc_out = nn.Linear(N1, 6)  # Combine outputs into a single layer
        self.denormalization = DenormalizationLayer(output_min, output_max)
        self.initialize_weights()

    def forward(self, x):
        x = self.normalization(x)
        x = self.fourier_embedding(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = F.gelu(self.fc5(x))
        outputs = self.fc_out(x)
        outputs = self.denormalization(outputs)
        u, v, p, k, omega, c = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4], outputs[:, 5]
        return [u, v, p, k, omega, c]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class NormalizationLayer(nn.Module):
    def __init__(self, input_min, input_max):
        super(NormalizationLayer, self).__init__()
        self.input_min = torch.tensor(input_min, dtype=torch.float32)
        self.input_max = torch.tensor(input_max, dtype=torch.float32)

    def forward(self, x):
        if self.input_min.device != x.device:
            self.input_min = self.input_min.to(x.device)
        if self.input_max.device != x.device:
            self.input_max = self.input_max.to(x.device)
        return (x - self.input_min) / (self.input_max - self.input_min)

class DenormalizationLayer(nn.Module):
    def __init__(self, output_min, output_max):
        super(DenormalizationLayer, self).__init__()
        self.output_min = torch.tensor(output_min, dtype=torch.float32)
        self.output_max = torch.tensor(output_max, dtype=torch.float32)

    def forward(self, x):
        if self.output_min.device != x.device:
            self.output_min = self.output_min.to(x.device)
        if self.output_max.device != x.device:
            self.output_max = self.output_max.to(x.device)
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

def smooth_maximum(a, b, alpha=10):
    b = b.expand_as(a)
    return torch.logsumexp(torch.stack([a, b], dim=0) * alpha, dim=0) / alpha

def smooth_minimum(a, b, alpha=10):
    b = b.expand_as(a)
    return -torch.logsumexp(torch.stack([-a, -b], dim=0) * alpha, dim=0) / alpha

def smooth_conditional(cond, true_val, false_val, alpha=10):
    true_val = true_val.expand_as(cond)
    false_val = false_val.expand_as(cond)
    return cond.sigmoid() * true_val + (1 - cond.sigmoid()) * false_val

def safe_sqrt(tensor, epsilon=1e-16):
    return torch.sqrt(tensor + epsilon)

# Define the PDE residuals for the SST k-omega model and convection-diffusion equation
def pde_residuals(model, x, y, t, Re, theta):
    U_star = mu_rho_l * Re

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
    u = u.view(-1, 1)
    v = v.view(-1, 1)
    p = p.view(-1, 1)
    k = k.view(-1, 1)
    omega = omega.view(-1, 1)
    c = c.view(-1, 1)

    # Clamping k and omega to avoid negative values and extreme values
    k = torch.clamp(k, min=1e-10, max=1e6)
    omega = torch.clamp(omega, min=1e-6, max=1e6)

    # Compute first-order derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    k_x = torch.autograd.grad(k, x, grad_outputs=torch.ones_like(k), create_graph=True)[0]
    k_y = torch.autograd.grad(k, y, grad_outputs=torch.ones_like(k), create_graph=True)[0]
    k_t = torch.autograd.grad(k, t, grad_outputs=torch.ones_like(k), create_graph=True)[0]
    omega_x = torch.autograd.grad(omega, x, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
    omega_y = torch.autograd.grad(omega, y, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
    omega_t = torch.autograd.grad(omega, t, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    y_hat = safe_sqrt(x ** 2 + y ** 2) - 40 / L_star  # non-dim(distance) - radius/L_star
    D_omega_plus = smooth_maximum((2 / (sigma_omega2 * omega)) * (k_x * omega_x + k_y * omega_y), torch.tensor(1e-10, device=x.device))

    eps = 1e-6
    phi_11 = safe_sqrt(k) / (0.09 * omega * y_hat+eps)
    phi_12 = 500 / (Re * y_hat ** 2 * omega+eps)
    phi_13 = 4 * k / (sigma_omega2 * D_omega_plus * y_hat ** 2+eps)
    phi_1 = smooth_minimum(smooth_maximum(phi_11, phi_12), phi_13)
    phi_21 = (2 * safe_sqrt(k)) / (0.09 * omega * y_hat+eps)
    phi_22 = 500 / (Re * y_hat ** 2 * omega+eps)
    phi_2 = smooth_maximum(phi_21, phi_22)

    # Clamping intermediate terms to avoid extreme values
    phi_11 = torch.clamp(phi_11, min=-1e10, max=1e6)
    phi_12 = torch.clamp(phi_12, min=-1e10, max=1e6)
    phi_13 = torch.clamp(phi_13, min=-1e10, max=1e6)
    phi_1 = torch.clamp(phi_1, min=-1e10, max=1e6)
    phi_21 = torch.clamp(phi_21, min=-1e10, max=1e6)
    phi_22 = torch.clamp(phi_22, min=-1e10, max=1e6)
    phi_2 = torch.clamp(phi_2, min=-1e10, max=1e6)

    dummy_1 = torch.autograd.grad(safe_sqrt(k), y, grad_outputs=torch.ones_like(k), create_graph=True)[0]

    F1 = torch.tanh(phi_1 ** 4)
    F2 = torch.tanh(phi_2)
    beta_i = F1 * beta_i1 + (1 - F1) * beta_i2
    alpha_star_0 = beta_i / 3
    alpha_infinity_1 = beta_i1 / beta_star_infinity - kappa ** 2 / (sigma_omega1 * (beta_star_infinity)**0.5)
    alpha_infinity_2 = beta_i2 / beta_star_infinity - kappa ** 2 / (sigma_omega2 * (beta_star_infinity)**0.5)
    alpha_infinity = F1 * alpha_infinity_1 + (1 - F1) * alpha_infinity_2

    Re_t = k / (mu * omega)
    alpha_star = alpha_star_infinity * (alpha_star_0 + Re_t / R_k) / (1 + Re_t / R_k)
    alpha = (alpha_infinity / alpha_star) * ((alpha_0 + Re_t / R_omega) / (1 + Re_t / R_omega))
    beta_star_i = beta_star_infinity * ((4 / 15 + (Re_t / R_beta) ** 4) / (1 + (Re_t / R_beta) ** 4))
    M_t = U_star * safe_sqrt(2 * k / (gamma * R * T))
    F_Mt = smooth_conditional(M_t <= M_t0, torch.zeros_like(M_t), M_t ** 2 - M_t0 ** 2)

    beta_star = beta_star_i * (1 + xi_star * F_Mt)
    beta = beta_i * (1 - beta_star_i / beta_i * xi_star * F_Mt)
    sigma_k = 1 / (F1 / sigma_k1 + (1 - F1) / sigma_k2)
    sigma_omega = 1 / (F1 / sigma_omega1 + (1 - F1) / sigma_omega2)
    S = safe_sqrt(2 * ((u_x) ** 2 + (v_y) ** 2 + 0.5 * (u_y + v_x) ** 2))

    mu_t = k / omega * (1 / smooth_maximum(1 / alpha_star, S * F2 / (a1 * omega)))
    mu_t = torch.clamp(mu_t, min=1e-10, max=1e6)

    G_k = mu_t * S ** 2
    Y_k = beta_star * k * omega
    G_k_tilde = smooth_minimum(G_k, 10 * beta_star * k * omega)

    G_omega = alpha / mu_t * G_k_tilde
    Y_omega = beta * omega ** 2
    D_omega = 2 * (1 - F1) * (sigma_omega2 / omega) * (k_x * omega_x + k_y * omega_y)

    continuity_residual = u_x + v_y
    x_mom_x = (1/Re + mu_t) * (4/3 * u_x - 2/3 * v_y)
    x_mom_y = (1/Re + mu_t) * (v_x + u_y)
    x_mom_gradx = torch.autograd.grad(x_mom_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_mom_grady = torch.autograd.grad(x_mom_y, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_momentum_residual = u_t + u * u_x + v * u_y + p_x - x_mom_gradx - x_mom_grady
    y_mom_grady = torch.autograd.grad((1/Re + mu_t) * (4/3 * v_y - 2/3 * u_x), y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_mom_gradx = torch.autograd.grad((1/Re + mu_t) * (v_x + u_y), x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_momentum_residual = v_t + u * v_x + v * v_y + p_y - y_mom_grady - y_mom_gradx

    k_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_x, x, grad_outputs=torch.ones_like(k_x), create_graph=True)[0]
    k_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_y, y, grad_outputs=torch.ones_like(k_y), create_graph=True)[0]

    k_residual = k_t + u * k_x + v * k_y - k_transport_term1 - k_transport_term2 - G_k + Y_k

    omega_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]

    omega_transport_term1 = torch.clamp(omega_transport_term1, min=-10.0, max=10.0)
    omega_transport_term2 = torch.clamp(omega_transport_term2, min=-10.0, max=10.0)
    G_omega = torch.clamp(G_omega, min=-10.0, max=10.0)
    Y_omega = torch.clamp(Y_omega, min=-10.0, max=10.0)
    D_omega = torch.clamp(D_omega, min=-10.0, max=10.0)

    omega_residual = omega_t + u * omega_x + v * omega_y - omega_transport_term1 - omega_transport_term2 - G_omega + Y_omega - D_omega
    c_residual = c_t + u * c_x + v * c_y - (1 / Re) * (c_x + c_y)  # Convection-diffusion equation

    pde_check_list = {
        'x': x,
        'y': y,
        't': t,
        'Re': Re,
        'theta': theta,
        'u': u,
        'v': v,
        'p': p,
        'k': k,
        'omega': omega,
        'c': c,
        'u_x': u_x,
        'u_y': u_y,
        'u_t': u_t,
        'v_x': v_x,
        'v_y': v_y,
        'v_t': v_t,
        'p_x': p_x,
        'p_y': p_y,
        'k_x': k_x,
        'k_y': k_y,
        'k_t': k_t,
        'omega_x': omega_x,
        'omega_y': omega_y,
        'omega_t': omega_t,
        'c_x': c_x,
        'c_y': c_y,
        'c_t': c_t,
        'y_hat': y_hat,
        'D_omega_plus': D_omega_plus,
        'phi_11': phi_11,
        'phi_12': phi_12,
        'phi_13': phi_13,
        'phi_1': phi_1,
        'phi_21': phi_21,
        'phi_22': phi_22,
        'phi_2': phi_2,
        'F1': F1,
        'F2': F2,
        'beta_i': beta_i,
        'alpha_star_0': alpha_star_0,
        'alpha_infinity': alpha_infinity,
        'Re_t': Re_t,
        'alpha_star': alpha_star,
        'alpha': alpha,
        'beta_star_i': beta_star_i,
        'M_t': M_t,
        'F_Mt': F_Mt,
        'beta_star': beta_star,
        'beta': beta,
        'sigma_k': sigma_k,
        'sigma_omega': sigma_omega,
        'S': S,
        'mu_t': mu_t,
        'G_k': G_k,
        'Y_k': Y_k,
        'G_k_tilde': G_k_tilde,
        'G_omega': G_omega,
        'Y_omega': Y_omega,
        'D_omega': D_omega,
        'continuity_residual': continuity_residual,
        'dummy_1': dummy_1,
        'x_mom_x': x_mom_x,
        'x_mom_y': x_mom_y,
        'x_mom_gradx': x_mom_gradx,
        'x_mom_grady': x_mom_grady,
        'x_momentum_residual': x_momentum_residual,
        'y_mom_grady': y_mom_grady,
        'y_mom_gradx': y_mom_gradx,
        'y_momentum_residual': y_momentum_residual,
        'k_transport_term1': k_transport_term1,
        'k_transport_term2': k_transport_term2,
        'k_residual': k_residual,
        'omega_transport_term1': omega_transport_term1,
        'omega_transport_term2': omega_transport_term2,
        'omega_residual': omega_residual,
        'c_residual': c_residual,
    }

    return continuity_residual, x_momentum_residual, y_momentum_residual, k_residual, omega_residual, c_residual

all_ones_weights = {
        'pde': torch.tensor([1.0] * 6, device=device),
        'bc': torch.tensor([1.0] * 14, device=device),
        'ic': torch.tensor([1.0] * 6, device=device),
        'sparse': torch.tensor([1.0] * 6, device=device)
    }

all_normalized_weights = {
    'pde': torch.tensor([0.2, 0.1, 0.1, 10.0, 0.01, 1e10], device=device),
    # inlet u, v, k, omega (all Dirichlet)
    'bc': torch.tensor([0.2, 1.0, 1.0, 0.01,
                        # symmetry_u,v,p,k,omega (all Neumann),v_Dirichlet
                        0.2, 0.2, 0.2, 100, 0.01, 0.2,
                        # out_p  wall_u,v,k (all Dirichlet)
                        0.1, 0.2, 0.1, 10.0], device=device),
    'ic': torch.tensor([0.2, 0.2, 0.2, 1000, 0.2, 1e10], device=device),
    'sparse': torch.tensor([0.2, 0.2, 0.2, 1000, 0.2, 1e10], device=device)
}

def update_weights(model, pde_inputs, boundary_conditions, initial_conditions, sparse_data, weights, writer, epoch):
    # Define the max allowable weight values and min value for each weight
    inputs_0, outputs_0 = initial_conditions
    max_weight = 1000.0
    max_weights = {
        'bc': torch.full((14,), max_weight, device=device),
        'ic': torch.full((6,), max_weight, device=device),
        'pde': torch.full((6,), max_weight, device=device),
        'sparse': torch.full((6,), max_weight, device=device)
    }

    min_weight = 1.0

#   criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # Calculate gradients for each loss component type
    def calculate_gradients(loss_components):
        gradients = []
        for loss in loss_components:
            loss.backward(retain_graph=True)
            grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))
            gradients.append(grad_norm.item())
            model.zero_grad()
        return gradients

    # Calculate loss components
    ic_losses = [criterion(model(torch.cat(inputs_0, dim=1))[i], outputs_0[i].squeeze()) for i in range(6)]

    bc_losses = bc_calc_loss(model, boundary_conditions, criterion)
    pde_losses = [
        criterion(residual, torch.zeros_like(residual))
        for residual in pde_residuals(model, *pde_inputs)
    ]

    inputs_sparse, outputs_sparse = prepare_inputs_outputs(sparse_data)

    sparse_losses = [criterion(model(torch.cat(inputs_sparse, dim=1))[i], outputs_sparse[i].squeeze()) for i in range(6)]

    normalized_ic_losses = [all_normalized_weights['ic'][i]*ic_losses[i]
                       for i in range(len(all_normalized_weights['ic']))]
    normalized_sparse_losses = [all_normalized_weights['sparse'][i]*sparse_losses[i]
                       for i in range(len(all_normalized_weights['sparse']))]
    normalized_pde_losses = [all_normalized_weights['pde'][i]*pde_losses[i]
                     for i in range(len(all_normalized_weights['pde']))]
    normalized_bc_losses = [all_normalized_weights['bc'][i]*bc_losses[i]
                    for i in range(len(all_normalized_weights['bc']))]

    # Calculate gradients
    gradients = {
        'ic': calculate_gradients(ic_losses),
        'bc': calculate_gradients(bc_losses),
        'pde': calculate_gradients(pde_losses),
        'sparse': calculate_gradients(sparse_losses)
    }

    total_norm = sum([sum(grads) for grads in gradients.values()])

    new_weights = {}

    # Update weights based on gradients, clamping if necessary
    eps_1=1e-6
    for key in weights.keys():
        weight_update = []
        for i, (grad, weight) in enumerate(zip(gradients[key], weights[key])):
            if weight != 0:
                updated_weight = total_norm / max(grad, eps_1)
                weight_update.append(updated_weight)
            else:
                weight_update.append(weight.item())
        weight_update = torch.tensor(weight_update, device=device)

        max_updated_weight = torch.max(weight_update)
        if max_updated_weight > 2*max_weight:
            scaling_factor = max_weight / max_updated_weight
            weight_update = weight_update * scaling_factor

        weight_update = torch.clamp(weight_update, min=min_weight)
        new_weights[key] = weight_update

    for key in new_weights:
        for i, w in enumerate(new_weights[key]):
            writer.add_scalar(f'weights/{key}_{i}', w, epoch)
    return new_weights

def bc_calc_loss(model, boundary_conditions, criterion):
    bc_losses = []
    for bc in boundary_conditions:
        inputs, conditions = bc
        x_b, y_b, t_b, Re_b, theta_b = inputs
        y_b.requires_grad_(True)

        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([x_b, y_b, t_b, Re_b, theta_b], dim=1))

        for variable, condition in conditions.items():
            if condition['type'] == 'Dirichlet':
                value = condition['value']
                if variable == 'u':
                    bc_losses.append(criterion(u_pred, value.squeeze()))
                elif variable == 'v':
                    bc_losses.append(criterion(v_pred, value.squeeze()))
                elif variable == 'p':
                    bc_losses.append(criterion(p_pred, value.squeeze()))
                elif variable == 'k':
                    bc_losses.append(criterion(k_pred, value.squeeze()))
                elif variable == 'omega':
                    bc_losses.append(criterion(omega_pred, value.squeeze()))
                elif variable == 'v_dir':
                    bc_losses.append(criterion(v_pred, value.squeeze()))

            elif condition['type'] == 'Neumann':
                dir_deriv = condition['dir_deriv']
                value = condition['value']
                if dir_deriv == 'y':
                    if variable == 'u':
                        deriv = torch.autograd.grad(u_pred, y_b, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
                    elif variable == 'v':
                        deriv = torch.autograd.grad(v_pred, y_b, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
                    elif variable == 'p':
                        deriv = torch.autograd.grad(p_pred, y_b, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
                    elif variable == 'k':
                        deriv = torch.autograd.grad(k_pred, y_b, grad_outputs=torch.ones_like(k_pred), create_graph=True)[0]
                    elif variable == 'omega':
                        deriv = torch.autograd.grad(omega_pred, y_b, grad_outputs=torch.ones_like(omega_pred), create_graph=True)[0]

                bc_losses.append(criterion(deriv, value))
    return bc_losses

class CustomDataset(Dataset):
    def __init__(self, data_array, num_intervals=10, num_simulations=5):
        self.data = torch.tensor(data_array, dtype=torch.float32).to(device)
        self.num_intervals = num_intervals
        self.num_simulations = num_simulations
        self.total_snapshots = 100
        self.snapshots_per_interval = 10
        self.total_data_size = 5*13001*100
        self.elements_per_snapshot = 5*13001
        self.elements_per_simulation = 13001

    def __len__(self):
        return self.total_data_size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_initial_condition_batch(self, batch_size):
        indices = torch.randperm(self.elements_per_snapshot)[:batch_size]
        return self.data[indices]

    def get_plotting_data(self, snapshot, simulation):
        start_idx = (snapshot - 1) * self.elements_per_snapshot + (simulation - 1) * self.elements_per_simulation
        end_idx = start_idx + self.elements_per_simulation
        return self.data[start_idx:end_idx]

    def get_data_from_interval(self, interval):
        start_snapshot = interval * self.snapshots_per_interval
        end_snapshot = (interval + 1) * self.snapshots_per_interval
        start_idx = start_snapshot * self.elements_per_snapshot
        end_idx = end_snapshot * self.elements_per_snapshot
        return self.data[start_idx:end_idx]

    def get_sparse_batch(self, batch_size):
        indices = torch.randperm(self.total_data_size)[:batch_size]
        return self.data[indices]

class IntervalDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_intervals = dataset.num_intervals

    def __iter__(self):
        for interval in range(self.num_intervals):
            interval_data = self.dataset.get_data_from_interval(interval)
            indices = torch.randperm(len(interval_data))[:self.batch_size]
            yield interval_data[indices]

    def __len__(self):
        # Number of intervals (one batch per interval)
        return self.num_intervals

def log_metrics(writer, tot_epoch, epoch, total_loss, ic_total_loss, bc_total_loss, sparse_total_loss, pde_total_loss,
                ic_losses, bc_losses, sparse_losses, pde_losses, weights, temporal_weights,
                model, dataset, lr):
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    reserved_memory = torch.cuda.memory_reserved() / 1024**2
    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)

    bc_names = [
        'inlet_u', 'inlet_v', 'inlet_k', 'inlet_omega',
        'symmetry_u', 'symmetry_v', 'symmetry_p', 'symmetry_k', 'symmetry_omega',
        'symmetry_v_dirichlet', 'outlet_p', 'wall_u', 'wall_v', 'wall_k' ]
    ic_names = ['u', 'v', 'p', 'k', 'omega', 'c']
    sparse_names = ['u', 'v', 'p', 'k', 'omega', 'c']
    pde_names = ['continuity', 'x_momentum', 'y_momentum', 'k_transport',
                      'omega_transport', 'convection_diffusion']

    writer.add_scalar('Memory/Allocated', allocated_memory, tot_epoch)
    writer.add_scalar('Memory/Reserved', reserved_memory, tot_epoch)
    writer.add_scalar('Gradients/Norm', grad_norm, tot_epoch)
    writer.add_scalar('LearningRate', lr, tot_epoch)
    writer.add_scalar('_total_loss', total_loss.item(), epoch)
    writer.add_scalar('bc/_total', bc_total_loss.item(), epoch)

    for i, bc_loss in enumerate(bc_losses):
        writer.add_scalar(f'bc/{bc_names[i]}', all_normalized_weights['bc'][i] * bc_loss.item(), epoch)

    writer.add_scalar('ic/_total', ic_total_loss.item(), epoch)
    for i, ic_loss in enumerate(ic_losses):
        writer.add_scalar(f'ic/{ic_names[i]}', all_normalized_weights['ic'][i] * ic_loss.item(), epoch)

    writer.add_scalar('sparse/_total', sparse_total_loss.item(), epoch)
    for i, sparse_loss in enumerate(sparse_losses):
        writer.add_scalar(f'sparse/{sparse_names[i]}', all_normalized_weights['sparse'][i] * sparse_loss.item(), epoch)

    writer.add_scalar('pde/_total', pde_total_loss.item(), epoch)
    for i, pde_loss in enumerate(pde_losses):
        writer.add_scalar(f'pde/{pde_names[i]}', all_normalized_weights['pde'][i] * pde_loss.item(), epoch)

    # Log weights
    for i in range(len(bc_losses)):
        writer.add_scalar(f'weights/bc/{bc_names[i]}', weights['bc'][i], epoch)
    for i in range(len(ic_losses)):
        writer.add_scalar(f'weights/ic/{ic_names[i]}', weights['ic'][i], epoch)
    for i in range(len(sparse_losses)):
        writer.add_scalar(f'weights/sparse/{sparse_names[i]}', weights['sparse'][i], epoch)
    for i in range(len(pde_losses)):
        writer.add_scalar(f'weights/pde/{pde_names[i]}', weights['pde'][i], epoch)

    # Log weights
    for j in range(10): # 10 is num_intervals
        for i in range(len(bc_losses)):
            writer.add_scalar(f'weights/i/bc/{bc_names[i]}', weights['bc'][i], epoch)
        for i in range(len(sparse_losses)):
            writer.add_scalar(f'weights/i/sparse/{sparse_names[i]}', weights['sparse'][i], epoch)
        for i in range(len(pde_losses)):
            writer.add_scalar(f'weights/i/pde/{pde_names[i]}', weights['pde'][i], epoch)

    # Log temporal weights
    if epoch % N_plot_tweight == 0:
        # Create a logarithmic plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yscale('log')

        eps2 = 0.2
        x_bc = np.array([i for i in range(10)])-eps2
        x_pde = np.array([i for i in range(10)])
        x_sparse = np.array([i for i in range(10)])+eps2
        # Plot all the values with different markers
        for i, obj in enumerate(temporal_weights):
            ax.scatter([x_bc[i]]*len(obj['bc']), obj['bc'].cpu(), color='green', marker='.', label='bc' if i == 0 else "")
            ax.scatter([x_pde[i]]*len(obj['pde']), obj['pde'].cpu(), color='blue', marker='.', label='pde' if i == 0 else "")
            ax.scatter([x_sparse[i]]*len(obj['sparse']), obj['sparse'].cpu(), color='red', marker='.', label='sparse' if i == 0 else "")

        # Add titles and labels
        ax.set_title('Temporal Weights')
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Temporal Weights (log scale)')
        ax.set_xticks(x_pde)  # Ensure all integers from 1 to 10 are included on the x-axis
        ax.legend()

        writer.add_figure('t_weights_epoch{epoch}', fig, epoch)

    if epoch == 1000:
# --- actuals ----------------------------------------------------------------
        snapshot, simulation = 1,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t1', U_star = 9.0)
        writer.add_figure('Actual t = 1s Fields', fig, epoch)

        snapshot, simulation = 2,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t2', U_star = 9.0)
        writer.add_figure('Actual t = 2s Fields', fig, epoch)

        snapshot, simulation = 4,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t4', U_star = 9.0)
        writer.add_figure('Actual t = 4s Fields', fig, epoch)

        snapshot, simulation = 8,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t8', U_star = 9.0)
        writer.add_figure('Actual t = 8s Fields', fig, epoch)

        snapshot, simulation = 16,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t16', U_star = 9.0)
        writer.add_figure('Actual t = 16s Fields', fig, epoch)

        snapshot, simulation = 32,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t32', U_star = 9.0)
        writer.add_figure('Actual t = 32s Fields', fig, epoch)

        snapshot, simulation = 64,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t64', U_star = 9.0)
        writer.add_figure('Actual t = 64s Fields', fig, epoch)

        snapshot, simulation = 100,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
        plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u,v,p,k,omega,c = plot_outputs
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_actual_t100', U_star = 9.0)
        writer.add_figure('Actual t = 100s Fields', fig, epoch)

    if epoch % N_plot_fields == 0:
# --- predictions ------------------------------------------------------------
        snapshot, simulation = 1,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t1_epoch{epoch}')
        writer.add_figure('Predicted t = 1s Fields', fig, epoch)

        snapshot, simulation = 2,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t2_epoch{epoch}')
        writer.add_figure('Predicted t = 2s Fields', fig, epoch)

        snapshot, simulation = 4,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t4_epoch{epoch}')
        writer.add_figure('Predicted t = 4s Fields', fig, epoch)

        snapshot, simulation = 8,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t8_epoch{epoch}')
        writer.add_figure('Predicted t = 8s Fields', fig, epoch)

        snapshot, simulation = 16,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t16_epoch{epoch}')
        writer.add_figure('Predicted t = 16s Fields', fig, epoch)

        snapshot, simulation = 32,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t32_epoch{epoch}')
        writer.add_figure('Predicted t = 32s Fields', fig, epoch)

        snapshot, simulation = 64,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t64_epoch{epoch}')
        writer.add_figure('Predicted t = 64s Fields', fig, epoch)

        snapshot, simulation = 100,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x,y,t,Re,theta = plot_inputs
        u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
        fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c27_pred_t100_epoch{epoch}')
        writer.add_figure('Predicted t = 100s Fields', fig, epoch)

    if epoch % N_loss_print == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

    if epoch % N_save_model == 0:
        save_model(model, 'c26_20K_u9.pth')

def plot_fields(x, y, u, v, p, k, omega, c, snapshot,
                simulation,  name = 'new_fig', U_star = None, save_dir="figures"):
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

        # Plotting
        kk = 1
        fig1 = plt.figure(figsize=(kk*18, kk*12))

        plt.subplot(3, 2, 1)
        plt.tricontourf(triang, u.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $u$ at time {snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.tricontourf(triang, v.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $v$ at time {snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, p.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $p$ at time {snapshot}s')
        plt.tight_layout()

        k_plot = k.squeeze()
        plt.subplot(3, 2, 2)
        plt.tricontourf(triang, k_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $k$ at time {snapshot}s')
        plt.tight_layout()

        omega_plot = omega.squeeze()
        plt.subplot(3, 2, 4)
        plt.tricontourf(triang, omega_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $\omega$ at time {snapshot}s')
        plt.tight_layout()

        c_plot = c.squeeze()
        plt.subplot(3, 2, 6)
        plt.tricontourf(triang, c_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $c$ at time {snapshot}s')
        plt.tight_layout()

        # Save the figure
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{name}_fields_t_{snapshot}_sim_{simulation}.png"))

        return fig1

def check_tensor_stats(tensor, name):
    if tensor.numel() == 0:
        print(f"{name} is empty")
    else:
        print(f"{name: <10} mean={tensor.mean().item(): .3e}, std={tensor.std().item(): .3e}, min={tensor.min().item(): .3e}, max={tensor.max().item(): .3e}")

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

def prepare_inputs_outputs(batch_data):
    inputs = [batch_data[:, i:i+1] for i in range(5)]
    outputs = [batch_data[:, i+5:i+6] for i in range(6)]
    return inputs, outputs

def calculate_ic_losses(model, inputs, outputs, criterion):
    concatenated_inputs = torch.cat(inputs, dim=1)
    predictions = model(concatenated_inputs)
    return [criterion(predictions[i], outputs[i].squeeze()) for i in range(6)]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    input_min = [-2.5, -2.5, 1, 30e6, 0]
    input_max = [7.5, 2.5, 100, 70e6, 2 * np.pi]
    output_min = [-1, -1, -2, 1e-8, 1e-8, 1e-8]
    output_max = [2, 1, 1, 0.1, 25, 5e-5]

    model = PINN(input_min, input_max, output_min, output_max).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=optim_config.learning_rate,
        betas=(optim_config.beta1, optim_config.beta2),
        eps=optim_config.eps,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_config.decay_steps, gamma=optim_config.decay_rate)
    writer = SummaryWriter(log_dir='runs_/c27_Jun26_01')

    weights = {
        'bc': torch.ones(14, device=device, requires_grad=True),
        'ic': torch.ones(6, device=device, requires_grad=True),
        'pde': torch.ones(6, device=device, requires_grad=True),
        'sparse': torch.ones(6, device=device, requires_grad=True)
    }

    run_schedule = [
        (50000, all_ones_weights),
    ]

    data_array = np.load("data/preprocessed_clipped.npy")
    dataset = CustomDataset(data_array, num_intervals=10, num_simulations=5)
    data_loader = IntervalDataLoader(dataset, batch_size=256)

    tot_epoch = 0

    for epochs, initial_weights in run_schedule:
        if initial_weights is not None:
            weights = initial_weights

        for epoch in range(epochs):
            total_loss = 0

            batch_size = 256
            batch_size_ic = batch_size
            ic_batch = dataset.get_initial_condition_batch(batch_size_ic).to(device)
            ic_inputs, ic_outputs = prepare_inputs_outputs(ic_batch)
            ic_losses = [criterion(model(torch.cat(ic_inputs, dim=1))[i], ic_outputs[i].squeeze()) for i in range(6)]
            ic_total_loss = sum(all_normalized_weights['ic'][i]*weights['ic'][i]*ic_losses[i]
                                    for i in range(len(weights['ic'])))

            cumulative_losses = {
                'pde': torch.zeros(6),
                'sparse': torch.zeros(6),
                'bc': torch.zeros(14)
            }

            log_temporal_weights = {
                'pde': torch.zeros(6),
                'sparse': torch.zeros(6),
                'bc': torch.zeros(14)
            }

            intervals_temporal_weights = []
            interval = 0
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                inputs, outputs = prepare_inputs_outputs(batch_data)
                x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse = inputs
                u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = outputs

                boundary_conditions = generate_boundary_conditions(interval, batch_size//4)
                pde_inputs = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse)


                raw_losses = {
                    'pde': torch.zeros(6),
                    'sparse': torch.zeros(6),
                    'bc': torch.zeros(14)
                }


                raw_losses['pde'] = [criterion(residual, torch.zeros_like(residual))
                              for residual in pde_residuals(model, *pde_inputs)]
                raw_losses['bc'] = bc_calc_loss(model, boundary_conditions, criterion)
                raw_losses['sparse'] = [criterion(model(torch.cat(inputs, dim=1))[i], outputs[i].squeeze()) for i in range(6) ]

                eps_ = 1
                temporal_weights = {key: torch.clamp(torch.exp(-eps_*all_normalized_weights[key].to(device)*cumulative_losses[key].to(device)), min=1e-2, max=None)
                    for key in ['bc', 'pde', 'sparse']}

                print(f'--- cumulative losses ----- interval {interval} --')
                print(all_normalized_weights['pde'].to(device)*cumulative_losses['pde'].to(device))
                print(f'--- temporal weights ------ interval {interval} --')
                print(temporal_weights['pde'])

                intervals_temporal_weights.append(temporal_weights)

                temporal_weighted_losses = {key: [loss * temporal_weights[key][i]
                  for i, loss in enumerate(raw_losses[key])] for key in ['bc', 'pde', 'sparse']}

                cumulative_losses = {key: cumulative_losses[key] + torch.tensor(
                    [loss.item() for loss in temporal_weighted_losses[key]]) for key in ['bc', 'pde', 'sparse']}
                interval += 1

            sparse_losses = [weights['sparse'][i] * cumulative_losses['sparse'][i] for i in range(len(weights['sparse']))]
            pde_losses = [weights['pde'][i] * cumulative_losses['pde'][i] for i in range(len(weights['pde']))]
            bc_losses = [weights['bc'][i] * cumulative_losses['bc'][i] for i in range(len(weights['bc']))]

            sparse_total_loss = sum(all_normalized_weights['sparse'][i]*sparse_losses[i]
                                    for i in range(len(all_normalized_weights['sparse'])))
            pde_total_loss = sum(all_normalized_weights['pde'][i]*pde_losses[i]
                                    for i in range(len(all_normalized_weights['pde'])))
            bc_total_loss = sum(all_normalized_weights['bc'][i]*bc_losses[i]
                                    for i in range(len(all_normalized_weights['bc'])))

            total_loss += ic_total_loss + sparse_total_loss + pde_total_loss + bc_total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            tot_epoch += 1

            if epoch % N_weight_update == 0 and epoch != 0:
                sparse_data = dataset.get_sparse_batch(batch_size).to(device)
                new_weights = update_weights(model, pde_inputs, boundary_conditions,\
                  (ic_inputs, ic_outputs), sparse_data, weights, writer, tot_epoch)

                alpha_ = 0.5
                for key in weights.keys():
                    weights[key] = alpha_ * weights[key] + (1 - alpha_) * new_weights[key]
                    #print(f"weights[{key}]:", [f"{weights[key][i].item():.1f}" for i in range(len(weights[key]))])

            if epoch % N_log_metrics == 0:
                log_metrics(writer, tot_epoch, epoch, total_loss, ic_total_loss,
                            bc_total_loss, sparse_total_loss, pde_total_loss,
                            ic_losses, cumulative_losses['bc'],
                            cumulative_losses['sparse'], cumulative_losses['pde'],
                            weights, intervals_temporal_weights, model, dataset, scheduler.get_last_lr()[0])

if __name__ == "__main__":
    main()
