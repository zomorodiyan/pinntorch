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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('run J23 ------------------------- 1 ---------------------------------')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
optim_config.decay_steps = 100
optim_config.grad_accum_steps = 0
optim_config.clip_norm = 2000.0
#optim_config.weight_decay = 1.0e-4

N_intervals = 10
t_start = 50

N_log_metrics = 10
N_loss_print = 10
N_weight_update = 10
N_plot_fields = 100
N_plot_tweight = 100
N_save_model = 500
N_plot_error = 1000
N_plot_residuals = 1000

def save_model(model, path):
    torch.save(model.state_dict(), path)

def generate_boundary_conditions(interval, num_samples):
    Re_medium = 1.225 * 9.0 * 80.0 / 1.7894e-5
    U_in = 1.0 # U_star ≡ U_in so the non-dim version is 1.0
    k_in_value = 3 / 2 * (0.05 * U_in) ** 2
    omega_in_value = 25.0 # if not cliped 2000*80/9

    t_low, t_high = t_start + interval * (10 // N_intervals), t_start + (interval + 1) * (10 // N_intervals)
    theta_values =  torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
                                 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi], device=device)
    N_each_bc = num_samples
    boundary_conditions = []
    x_in = torch.full((N_each_bc, 1), -200.0 / L_star).float().to(device)
    y_in = torch.linspace(-200 / L_star, 200 / L_star, N_each_bc).view(-1, 1).float().to(device)
    t_in = torch.randint(t_low, t_high, x_in.size()).float().to(device)
    Re_in = torch.full((N_each_bc, 1), Re_medium).float().to(device)
    theta_in = theta_values[torch.randint(len(theta_values), x_in.size(), device=device)].float()
    inputs_in = (x_in, y_in, t_in, Re_in, theta_in)
    conditions_in = {
        'u': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), 1.0).float().to(device)},
        'v': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).float().to(device)},
        'k': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), k_in_value).float().to(device)},
        'omega': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), omega_in_value).float().to(device)}
    }

    x_sym = torch.linspace(-200 / L_star, 600 / L_star, N_each_bc).view(-1, 1).float().to(device)
    y_sym = torch.where(torch.randint(0, 2, (N_each_bc, 1), device=device) == 0, -200.0, 200.0).float() / L_star
    t_sym = torch.randint(t_low, t_high, x_sym.size()).float().to(device)
    Re_sym = torch.full((N_each_bc, 1), Re_medium).float().to(device)
    theta_sym = theta_values[torch.randint(len(theta_values), x_sym.size(), device=device)].float()
    inputs_sym = (x_sym, y_sym, t_sym, Re_sym, theta_sym)
    conditions_sym = {
        'u': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).float().to(device)},
        'v': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).float().to(device)},
        'p': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).float().to(device)},
        'k': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).float().to(device)},
        'omega': {'type': 'Neumann', 'dir_deriv': 'y', 'value': torch.zeros_like(x_sym).float().to(device)},
        'v_dir': {'type': 'Dirichlet', 'value': torch.zeros_like(x_sym).float().to(device)}
    }

    x_out = torch.full((N_each_bc, 1), 600.0 / L_star).float().to(device)
    y_out = torch.linspace(-200 / L_star, 200 / L_star, N_each_bc).view(-1, 1).float().to(device)
    t_out = torch.randint(t_low, t_high, x_out.size()).float().to(device)
    Re_out = torch.full((N_each_bc, 1), Re_medium).float().to(device)
    theta_out = theta_values[torch.randint(len(theta_values), x_out.size(), device=device)].float()
    inputs_out = (x_out, y_out, t_out, Re_out, theta_out)
    conditions_out = {
        'p': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).float().to(device)}
    }

    theta_rand = torch.linspace(0, 2 * np.pi, N_each_bc).float().to(device)
    x_wall = (40 / L_star * torch.cos(theta_rand)).view(-1, 1).float().to(device)
    y_wall = (40 / L_star * torch.sin(theta_rand)).view(-1, 1).float().to(device)
    t_wall = torch.randint(t_low, t_high, x_wall.size()).float().to(device)
    Re_wall = torch.full((N_each_bc, 1), Re_medium).float().to(device)
    theta_wall = theta_values[torch.randint(len(theta_values), x_wall.size(), device=device)].float()
    inputs_wall = (x_wall, y_wall, t_wall, Re_wall, theta_wall)
    conditions_wall = {
#       'u': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).float().to(device)},
#       'v': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).float().to(device)},
        'k': {'type': 'Dirichlet', 'value': torch.zeros_like(x_wall).float().to(device)}
    }

    boundary_conditions.append((inputs_in, conditions_in))
    boundary_conditions.append((inputs_sym, conditions_sym))
    boundary_conditions.append((inputs_out, conditions_out))
    boundary_conditions.append((inputs_wall, conditions_wall))

    return boundary_conditions

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
    k = torch.clamp(k, min=1e-12, max=1e6)
    omega = torch.clamp(omega, min=1e-12, max=1e6)

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
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    y_hat = safe_sqrt(x ** 2 + y ** 2) - 40 / L_star  # non-dim(distance) - non-dim(radius)
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
    phi_11 = torch.clamp(phi_11, min=-1e12, max=1e6)
    phi_12 = torch.clamp(phi_12, min=-1e12, max=1e6)
    phi_13 = torch.clamp(phi_13, min=-1e12, max=1e6)
    phi_1 = torch.clamp(phi_1, min=-1e12, max=1e6)
    phi_21 = torch.clamp(phi_21, min=-1e12, max=1e6)
    phi_22 = torch.clamp(phi_22, min=-1e12, max=1e6)
    phi_2 = torch.clamp(phi_2, min=-1e12, max=1e6)

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
    mu_t = torch.clamp(mu_t, min=1e-12, max=1e6)

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
    D_t =  1/Re + (1/Re + mu_t)/(0.803)
    c_residual_term1 = torch.autograd.grad(D_t*c_x-u*c, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_residual_term2 = torch.autograd.grad(D_t*c_y-v*c, y, grad_outputs=torch.ones_like(c_y), create_graph=True)[0]

    c_residual = c_t - c_residual_term1 - c_residual_term2

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
        'c_xx': c_xx,
        'c_yy': c_yy,
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
        'bc': torch.tensor([1.0] * 12, device=device),
        'ic': torch.tensor([1.0] * 6, device=device),
        'sparse': torch.tensor([1.0] * 6, device=device)
    }

all_normalized_weights = {
    'pde': torch.tensor([0.2, 0.1, 0.1, 100.0, 0.0001, 1e8], device=device),
    # inlet u, v, k, omega (all Dirichlet)
    'bc': torch.tensor([0.2, 1.0, 1.0, 0.001,
                        # symmetry_u,v,p,k,omega (all Neumann),v_Dirichlet
                        0.2, 0.2, 0.2, 10, 0.01, 0.2,
                        # out_p  wall_k (all Dirichlet)
                        0.1, 100.0], device=device),
    'ic': torch.tensor([0.2, 0.2, 0.2, 100, 0.01, 1e10], device=device),
    'sparse': torch.tensor([0.2, 0.2, 0.2, 100, 0.01, 1e10], device=device)
}

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
        self.total_snapshots = 10
        self.snapshots_per_interval = 1
        self.elements_per_simulation = 13001
        self.total_data_size = self.num_simulations*self.elements_per_simulation*self.num_intervals
        self.elements_per_snapshot = self.num_simulations*self.elements_per_simulation

    def __len__(self):
        return self.total_data_size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_initial_condition_batch(self, batch_size):
        indices = torch.randperm(self.elements_per_snapshot)[:batch_size] + self.elements_per_simulation * self.num_simulations * t_start
        return self.data[indices]

    def get_plotting_data(self, snapshot, simulation):
        start_idx = (snapshot - 1) * self.elements_per_snapshot + (simulation - 1) * self.elements_per_simulation + self.elements_per_simulation * self.num_simulations * t_start
        end_idx = start_idx + self.elements_per_simulation
        return self.data[start_idx:end_idx]

    def get_data_from_interval(self, interval):
        start_snapshot = (interval + t_start) * self.snapshots_per_interval
        end_snapshot = (interval + t_start+1) * self.snapshots_per_interval
        start_idx = start_snapshot * self.elements_per_snapshot
        end_idx = end_snapshot * self.elements_per_snapshot
        return self.data[start_idx:end_idx]

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

    print('log metrics inside method')
    allocated_memory = torch.cuda.memory_allocated() / 1024**2
    reserved_memory = torch.cuda.memory_reserved() / 1024**2
    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)

    bc_names = [
        'in_u', 'in_v', 'in_k', 'in_$\omega$',
        'symm_u', 'symm_v', 'symm_p', 'symm_k', 'symm_$\omega$',
        'symm_v_D', 'out_p','wall_k' ]
    bc_group_names = [ 'inlet', 'symmetry', 'outlet', 'wall']
    ic_names = ['u', 'v', 'p', 'k', '$\omega$', 'c']
    sparse_names = ['u', 'v', 'p', 'k', '$\omega$', 'c']
    pde_names = ['continuity', 'x_momentum', 'y_momentum', 'k_transport',
                      '$\omega$_transprt', 'conv_diff']

    writer.add_scalar('Memory/Allocated', allocated_memory, tot_epoch)
    writer.add_scalar('Memory/Reserved', reserved_memory, tot_epoch)
    writer.add_scalar('Gradients/Norm', grad_norm, tot_epoch)
    writer.add_scalar('LearningRate', lr, tot_epoch)
    writer.add_scalar('_total_loss', total_loss, epoch)
    writer.add_scalar('bc/_total', bc_total_loss, epoch)
    writer.add_scalar('ic/_total', ic_total_loss.item(), epoch)
    writer.add_scalar('sparse/_total', sparse_total_loss.item(), epoch)
    writer.add_scalar('pde/_total', pde_total_loss.item(), epoch)

    for i, bc_loss in enumerate(bc_losses):
        writer.add_scalar(f'bc/{bc_names[i]}', bc_loss, epoch)
    for i, ic_loss in enumerate(ic_losses):
        writer.add_scalar(f'ic/{ic_names[i]}', ic_loss, epoch)
    for i, sparse_loss in enumerate(sparse_losses):
        writer.add_scalar(f'sparse/{sparse_names[i]}', sparse_loss, epoch)
    for i, pde_loss in enumerate(pde_losses):
        writer.add_scalar(f'pde/{pde_names[i]}', pde_loss, epoch)

    # Log weights
    for i in range(len(bc_losses)):
        writer.add_scalar(f'w_bc/{bc_names[i]}', weights['bc'][i], epoch)
    for i in range(len(ic_losses)):
        writer.add_scalar(f'w_ic/{ic_names[i]}', weights['ic'][i], epoch)
    for i in range(len(sparse_losses)):
        writer.add_scalar(f'w_sparse/{sparse_names[i]}', weights['sparse'][i], epoch)
    for i in range(len(pde_losses)):
        writer.add_scalar(f'w_pde/{pde_names[i]}', weights['pde'][i], epoch)

    # Log temporal weights
    if epoch % N_plot_tweight == 0:
        specified_colors = ['#E57373', '#FFB74D', '#BA68C8', '#64B5F6', '#4DB6AC', '#A9A9A9']
        def get_bc_color(index):
            if index < 4:
                return specified_colors[0]
            elif index < 10:
                return specified_colors[1]
            elif index == 10:
                return specified_colors[2]
            else:
                return specified_colors[3]

        # Create a logarithmic plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_yscale('log')

        # generate for x_axis
        eps2 = 0.2
        x_bc = np.array([i for i in range(10)]) - eps2
        x_pde = np.array([i for i in range(10)])
        x_sparse = np.array([i for i in range(10)]) + eps2

        for interval in range(10):
            # bc markers
            for i in range(12):
                if i <= 3:
                    color_index = 0
                elif 4 <= i <= 9:
                    color_index = 1
                elif i == 10:
                    color_index = 2
                else:
                    color_index = 3
                ax.scatter([x_bc[interval]], temporal_weights['bc'][interval][i].cpu().detach().numpy(),
                  color=specified_colors[color_index % len(specified_colors)],\
                    marker='^', edgecolors='black', label=f'bc_{bc_group_names[color_index]}' if interval == 0 and i in {0,4,10,11} else "")

            # pde markers
            for i in range(6):
                ax.scatter([x_pde[interval]], temporal_weights['pde'][interval][i].cpu().detach().numpy(),
                  color=specified_colors[i % len(specified_colors)], marker='o',\
                    edgecolors='black', label=f'{pde_names[i]}' if interval == 0 else "")

            # sparse markers
            for i in range(6):
                ax.scatter([x_sparse[interval]], temporal_weights['sparse'][interval][i].cpu().detach().numpy(),
                  color=specified_colors[i % len(specified_colors)], marker='s',\
                    edgecolors='black', label=f'sparse_{sparse_names[i]}' if interval == 0 else "")

        # Add titles and labels
        ax.set_title(f'Temporal Weights from {t_start+1}s to {t_start+10}s')
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Temporal Weights (log scale)')
        ax.set_xticks(x_pde)  # Ensure all integers from 1 to 10 are included on the x-axis
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), ncol=2)  # Set legend to two columns

        writer.add_figure('t_weights_epoch{epoch}', fig, epoch)

    if epoch == 1000:
# --- actuals ----------------------------------------------------------------
        for i in range(10):
            snapshot, simulation = i+1,1
            plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
            plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
            x,y,t,Re,theta = plot_inputs
            u,v,p,k,omega,c = plot_outputs
            fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c35_actual_t{t_start+i}', U_star = 9.0)
            writer.add_figure(f'Actual t = +1s Fields', fig, epoch)

    if epoch % N_plot_residuals== 0:
        snapshot, simulation = 1,1
        plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
        plot_inputs, _ = prepare_inputs_outputs(plot_data)
        x, y, _, _, _ = plot_inputs
        continuity, x_mom, y_mom, k_tr, omega_tr, conv_diff =\
          pde_residuals(model, *plot_inputs)
        fig = plot_residuals(x,y,continuity, x_mom, y_mom, k_tr, omega_tr, conv_diff,
          snapshot, simulation, f'c35_resi_t{t_start+1}_epoch{epoch}', U_star = 9.0)
        writer.add_figure('Residuals t = +1s', fig, epoch)

    if epoch % N_plot_error == 0:
        with torch.no_grad():
            snapshot, simulation = 1,1
            plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation)
            plot_inputs, plot_outputs = prepare_inputs_outputs(plot_data)
            x,y,t,Re,theta = plot_inputs
            u,v,p,k,omega,c = plot_outputs
            u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred =\
              model(torch.cat([x, y, t, Re, theta], dim=1))
            up = u_pred.unsqueeze(1) - u
            vp = v_pred.unsqueeze(1) - v
            pp = p_pred.unsqueeze(1) - p
            kp = k_pred.unsqueeze(1) - k
            op = omega_pred.unsqueeze(1) - omega
            cp = c_pred.unsqueeze(1) - c
            fig = plot_fields(x,y,up,vp,pp,kp,op,cp, snapshot, simulation, f'c35_errors_t{t_start+1}', U_star = 9.0)
            writer.add_figure('Errors t = +1s Fields', fig, epoch)

    if epoch % N_plot_fields == 0:
# --- predictions ------------------------------------------------------------
        with torch.no_grad():
            for i in range(10):
                snapshot, simulation = i+1,1
                plot_data = dataset.get_plotting_data(snapshot=snapshot, simulation=simulation).to(device)
                plot_inputs, _ = prepare_inputs_outputs(plot_data)
                x,y,t,Re,theta = plot_inputs
                u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))
                fig = plot_fields(x,y,u,v,p,k,omega,c, snapshot, simulation, f'c35_t{t_start+i+1}_epoch{epoch}', U_star = 9.0)
                writer.add_figure(f'Predicted t = +{i+1}s Fields', fig, epoch)

    if epoch % N_loss_print == 0:
        print(f'Epoch {epoch}, Loss: {total_loss}')

    if epoch % N_save_model == 0 and epoch != 0:
        save_model(model, f'../c35_{t_start+1}_{t_start+10}.pth')

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
        plt.title(f'Predicted $u$ at time {t_start+snapshot}s [$m/s$]')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.tricontourf(triang, v.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $v$ at time {t_start+snapshot}s [$m/s$]')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, p.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $p$ at time {t_start+snapshot}s [pa]')
        plt.tight_layout()

        k_plot = k.squeeze()
        plt.subplot(3, 2, 2)
        plt.tricontourf(triang, k_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $k$ at time {t_start+snapshot}s [$m^2/s^2$]')
        plt.tight_layout()

        omega_plot = omega.squeeze()
        plt.subplot(3, 2, 4)
        plt.tricontourf(triang, omega_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $\omega$ at time {t_start+snapshot}s [$1/s$]')
        plt.tight_layout()

        c_plot = c.squeeze()
        plt.subplot(3, 2, 6)
        plt.tricontourf(triang, c_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Predicted $c$ at time {t_start+snapshot}s []')
        plt.tight_layout()

        # Save the figure
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{name}_sim_$\theta$=0.png"))

        return fig1

def plot_residuals(x, y, x_mom, y_mom, continuity, k_tr, omega_tr, conv_diff, snapshot, simulation,  name = 'new_fig', save_dir="figures", U_star = None):
    with torch.no_grad():
        L_star = 80.0

        # Convert residuals to numpy arrays for plotting
        x_mom = x_mom.cpu().numpy()
        y_mom = y_mom.cpu().numpy()
        continuity = continuity.cpu().numpy()
        k_tr = k_tr.cpu().numpy()
        omega_tr = omega_tr.cpu().numpy()
        conv_diff = conv_diff.cpu().numpy()
        x = x.cpu()
        y = y.cpu()

        radius = 40.0 / L_star

        # dimensionalize the pde residuals if U_star is provided
        if U_star is not None:
            x = x * L_star
            y = y * L_star
            x_mom = x_mom * rho * U_star**2 / L_star
            y_mom = y_mom * rho * U_star**2 / L_star
            continuity = continuity * rho * U_star / L_star
            k_tr = k_tr * rho * U_star**3 / L_star
            omega_tr = omega_tr * U_star**2 / L_star**2
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
        plt.tricontourf(triang, x_mom.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'x-momentum residuals at time {t_start+snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.tricontourf(triang, y_mom.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'y-momentum residuals at time {t_start+snapshot}s ')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, continuity.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Continuity residuals at time {t_start+snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.tricontourf(triang, k_tr.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'$k$ residuals at time {t_start+snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.tricontourf(triang, omega_tr.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'$\omega$-transport residuals time {t_start+snapshot}s')
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.tricontourf(triang, conv_diff.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title(f'Convection-Diffusion residuals at time {t_start+snapshot}s')
        plt.tight_layout()

        # Save the figure
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{name}_sim_$\theta$=0.png"))
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
    print('main')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss().cuda()
    model = PINN().to(device)

    model.load_state_dict(torch.load(f'../models/c35_{t_start+1+10}_{t_start+10+10}.pth'))

    optimizer = optim.Adam(
        model.parameters(),
        lr=optim_config.learning_rate,
        betas=(optim_config.beta1, optim_config.beta2),
        eps=optim_config.eps,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_config.decay_steps, gamma=optim_config.decay_rate)
    writer = SummaryWriter(log_dir=f'../runs/c35_{t_start+1}_{t_start+10}_j8')

    weights = {
        'bc': torch.ones(12, device=device),
        'ic': torch.ones(6, device=device),
        'pde': torch.ones(6, device=device),
        'sparse': torch.ones(6, device=device)
    }

    print('scheduler')
    run_schedule = [
        (25000, all_ones_weights),
    ]

    batch_size = 128
    data_array = np.load("../data/preprocessed_clipped.npy")
    dataset = CustomDataset(data_array, num_intervals=10, num_simulations=5)
    data_loader = IntervalDataLoader(dataset, batch_size=batch_size)

    tot_epoch = 0

    for epochs, initial_weights in run_schedule:
        if initial_weights is not None:
            weights = initial_weights

        for epoch in range(epochs):
            print(f'epoch {epoch}')
            total_loss = 0.0
            ic_batch = dataset.get_initial_condition_batch(batch_size).to(device)
            ic_inputs, ic_outputs = prepare_inputs_outputs(ic_batch)
            raw_ic_losses = torch.stack([criterion(model(torch.cat(ic_inputs,
              dim=1))[i], ic_outputs[i].squeeze()) for i in range(6)])

            num_intervals = 10
            num_components_pde = 6
            num_components_bc = 12
            num_components_sparse = 6

# Initialize lists to temporarily store losses
            all_raw_losses_list = {
                'pde': [],
                'sparse': [],
                'bc': []
            }

# Loop over batches and intervals
            for interval, batch_data in enumerate(data_loader):
                #batch_data = batch_data.to(device)
                inputs, outputs = prepare_inputs_outputs(batch_data)
                x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse = [x.float() for x in inputs]
                u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = [y.float() for y in outputs]
                pde_inputs = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse)

                boundary_conditions = generate_boundary_conditions(interval, batch_size//2)

                raw_losses = {
                    'pde': [],
                    'sparse': [],
                    'bc': []
                }

                raw_losses['bc'] = [loss for loss in bc_calc_loss(model, boundary_conditions, criterion)]
                raw_losses['pde'] = [criterion(residual, torch.zeros_like(residual))
                                     for residual in pde_residuals(model, *pde_inputs)]
                raw_losses['sparse'] = [criterion(model(torch.cat(inputs, dim=1))[i], outputs[i].squeeze()) for i in range(6)]

                # Append raw losses to lists
                all_raw_losses_list['pde'].append(torch.stack(raw_losses['pde']))
                all_raw_losses_list['bc'].append(torch.stack(raw_losses['bc']))
                all_raw_losses_list['sparse'].append(torch.stack(raw_losses['sparse']))

# Convert lists to tensors
            all_raw_losses = {
                'pde': torch.stack(all_raw_losses_list['pde']),
                'sparse': torch.stack(all_raw_losses_list['sparse']),
                'bc': torch.stack(all_raw_losses_list['bc'])
            }

# Calculate temporal weights
            eps_ = 10
            temporal_weights = {
                key: torch.ones((10, len(weights[key])), device=device)
                for key in ['bc', 'pde', 'sparse']
            }

            for interval in range(1, num_intervals):
                for key in ['bc', 'pde', 'sparse']:
                    temporal_weights[key][interval] = torch.clamp(
                        torch.exp(-eps_ * all_normalized_weights[key] * torch.cumsum(all_raw_losses[key][:interval], dim=0)[-1]),
                        min=1e-3, max=None
                    )

# Calculate temporal weighted losses
            temporal_weighted_losses = {
                key: torch.ones((10, len(weights[key])), device=device)
                for key in ['bc', 'pde', 'sparse']
            }

            for interval in range(num_intervals):
                for key in ['bc', 'pde', 'sparse']:
                    temporal_weighted_losses[key][interval] = all_raw_losses[key][interval] * temporal_weights[key][interval]

            sum_temporal_weighted_losses = {
                key: temporal_weighted_losses[key].sum(dim=0)
                for key in ['bc', 'pde', 'sparse']
            }

            normalized_raw_losses = {
                'ic': [all_normalized_weights['ic'][i] * raw_ic_losses[i]for i in range(6)],
                'bc': [all_normalized_weights['bc'][i] * sum_temporal_weighted_losses['bc'][i] for i in range(12)],
                'sparse': [all_normalized_weights['sparse'][i] * sum_temporal_weighted_losses['sparse'][i] for i in range(6)],
                'pde': [all_normalized_weights['pde'][i] * sum_temporal_weighted_losses['pde'][i] for i in range(6)],
            }

            weighted_losses = {
                'ic': weights['ic'] * raw_ic_losses,
                'bc': weights['bc'] * sum_temporal_weighted_losses['bc'],
                'sparse': weights['sparse'] * sum_temporal_weighted_losses['sparse'],
                'pde': weights['pde'] * sum_temporal_weighted_losses['pde']
            }

            normalized_losses = {
                'ic': all_normalized_weights['ic'] * weighted_losses['ic'],
                'bc':  all_normalized_weights['sparse'] * weighted_losses['sparse'],
                'sparse': all_normalized_weights['pde'] * weighted_losses['pde'],
                'pde': all_normalized_weights['bc'] * weighted_losses['bc'],
            }

#--- update global weights ---------------------------------------------------
            if epoch % N_weight_update == 0 and epoch != 0:
                max_weight = 1000.0

                gradient_norms = {
                    'ic': torch.zeros(6, device=device),
                    'pde': torch.zeros(6, device=device),
                    'sparse': torch.zeros(6, device=device),
                    'bc': torch.zeros(12, device=device)
                }

                min_bc = 1.0
                min_ic = 50.0
                min_sparse = 50.0
                min_pde = 1.0
                min_weights = {
                    'ic': torch.tensor([min_ic, min_ic, min_ic, min_ic, min_ic, 2*min_ic], device=device),
                    'pde': torch.tensor([min_pde, min_pde, min_pde, 5*min_pde, 50*min_pde, 50*min_pde], device=device),
                    # inlet u, v, k, omega (all Dirichlet)
                    'bc': torch.tensor([min_bc, min_bc, min_bc, min_bc,
                                        # symmetry_u,v,p,k,omega (all Neumann),v_Dirichlet
                                        min_bc, min_bc, min_bc, min_bc, min_bc, min_bc,
                                        # out_p  wall_u,v,k (all Dirichlet)
                                        min_bc, 5*min_bc], device=device),
                    'sparse': torch.tensor([min_sparse, min_sparse, min_sparse,
                    min_sparse, min_sparse, 2*min_sparse], device=device)
                }

                for key in ['ic', 'bc', 'pde', 'sparse']:
                    for i in range(len(weights[key])):
                        model.zero_grad()
                        normalized_raw_losses[key][i].backward(retain_graph=True)  # Compute gradients

                        # Calculate gradient norm
                        grad_norm = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.norm().item() ** 2
                        grad_norm = grad_norm ** 0.5

                        gradient_norms[key][i] = grad_norm

                total_norm = sum([sum(grads) for grads in gradient_norms.values()])


                for key in ['ic', 'bc', 'pde', 'sparse']:
                    weight_update = []
                    eps_1 = 1e-6
                    for i, (grad, weight) in enumerate(zip(gradient_norms[key], weights[key])):
                        if weight != 0:
                            updated_weight = total_norm / max(grad, eps_1)
                            weight_update.append(updated_weight)
                        else:
                            weight_update.append(weight.item())
                    if len(weight_update) == 0:
                        print(f"No gradients found for {key}, skipping weight update")
                        continue

                    weight_update = torch.tensor(weight_update, device=device)

                    max_updated_weight = torch.max(weight_update)
                    if max_updated_weight > 2 * max_weight:
                        scaling_factor = max_weight / max_updated_weight
                        weight_update = weight_update * scaling_factor

                    weight_update = torch.max(weight_update, min_weights[key])
                    weight_update = torch.clamp(weight_update, max=max_weight)

                    alpha_ = 0.5
                    weights[key] = alpha_ * weights[key] + (1 - alpha_) * weight_update
                    print(f"weights[{key}]:", [f"{weights[key][i].item():.1f}" for i in range(len(weights[key]))])

            # Sum the normalized losses
            ic_total_loss = sum(normalized_losses['ic'])
            sparse_total_loss = sum(normalized_losses['sparse'])
            pde_total_loss = sum(normalized_losses['pde'])
            bc_total_loss = sum(normalized_losses['bc'])

            total_loss += ic_total_loss + sparse_total_loss + pde_total_loss + bc_total_loss

            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_config.clip_norm)
            optimizer.step()
            scheduler.step()
            tot_epoch += 1

#--- log metrics (tensorboard) -----------------------------------------------
            if epoch % N_log_metrics == 0:
                print('log metrics in main')
                log_metrics(writer, tot_epoch, epoch, total_loss, ic_total_loss,
                            bc_total_loss, sparse_total_loss, pde_total_loss,
                            [normalized_raw_losses['ic'][i].item() for i in range(6)],
                            [normalized_raw_losses['bc'][i].item() for i in range(12)],
                            [normalized_raw_losses['sparse'][i].item() for i in range(6)],
                            [normalized_raw_losses['pde'][i].item() for i in range(6)],
                            weights, temporal_weights, model, dataset, scheduler.get_last_lr()[0])

if __name__ == "__main__":
    main()
