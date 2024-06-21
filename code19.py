import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from utils import calculate_range_and_divergence, check_for_nan_and_inf
from utils import formatted_print, plot_fields, check_tensor_stats
from ml_collections import ConfigDict
from torch.profiler import profile, record_function, ProfilerActivity

print('run J20-19')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_seed(seed):
    import torch
    import numpy as np
    import random

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
optim_config.decay_steps = 2000
optim_config.grad_accum_steps = 0
optim_config.clip_norm = 10000.0
optim_config.weight_decay = 1.0e-4

N_bc = 100
N_ic = 100
N_pde = 100
N_sparse = 100

N_loss_print = 1
N_save_model = 1000
N_weight_update = 10
N_plot_fields = 100


def generate_boundary_conditions(num_samples, interval):
    Re_medium = 1.225 * 9.0 * 80.0 / 1.7894e-5
    k_in_value = 3 / 2 * (0.05 * 9.0) ** 2 / 9.0 ** 2
    omega_in_value = 30.0

    N_each_bc = N_bc
    boundary_conditions = []

    x_in = torch.full((N_each_bc, 1), -200.0 / L_star).to(device)
    y_in = torch.linspace(-200 / L_star, 200 / L_star, N_each_bc).view(-1, 1).to(device)
    t_in = torch.full_like(x_in, interval).float().to(device)
    Re_in = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_in = torch.zeros_like(x_in).to(device)
    inputs_in = (x_in, y_in, t_in, Re_in, theta_in)
    conditions_in = {
        'u': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), 1.0).to(device)},
        'v': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).to(device)},
        'k': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), k_in_value).to(device)},
        'omega': {'type': 'Dirichlet', 'value': torch.full((N_each_bc, 1), omega_in_value).to(device)}
    }

    x_sym = torch.linspace(-200 / L_star, 600 / L_star, N_each_bc).view(-1, 1).to(device)
    y_sym = (torch.where(torch.randint(0, 2, (N_each_bc, 1), device=device) == 0, -200.0, 200.0) / L_star).to(device)
    t_sym = torch.full_like(x_sym, interval).float().to(device)
    Re_sym = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_sym = torch.zeros_like(x_sym).to(device)
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
    t_out = torch.full_like(x_out, interval).float().to(device)
    Re_out = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_out = torch.zeros_like(x_out).to(device)
    inputs_out = (x_out, y_out, t_out, Re_out, theta_out)
    conditions_out = {
        'p': {'type': 'Dirichlet', 'value': torch.zeros((N_each_bc, 1)).to(device)}
    }

    theta_rand = torch.linspace(0, 2 * np.pi, N_each_bc).to(device)
    x_wall = (40 / L_star * torch.cos(theta_rand)).view(-1, 1).to(device)
    y_wall = (40 / L_star * torch.sin(theta_rand)).view(-1, 1).to(device)
    t_wall = torch.full_like(x_wall, interval).float().to(device)
    Re_wall = torch.full((N_each_bc, 1), Re_medium).to(device)
    theta_wall = torch.zeros_like(x_wall).to(device)
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
only_bc_weights = {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([1.0] * 14, device=device),
        'ic': torch.tensor([0.0] * 6, device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }
bc_ic_weights = {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([1.0] * 14, device=device),
        'ic': torch.tensor([1.0] * 6, device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }
bc_ic_normalized_weights = {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([3.0**-2, 2.0**-2, 0.1**-2,(30.0)**-2, (3.0*20)**-2, (2.0*20)**-2, (3.0*20)**-2,
                            (0.1*20)**-2, (30*20)**-2, 2.0**-2, 3.0**-2, 3.0**-2, 2.0**-2, 0.1**-2], device=device),
        'ic': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2, 0.01**-2], device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }
bc_normalized_weights = {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([3.0**-2, 2.0**-2, 0.1**-2,(30.0)**-2, (3.0*20)**-2, (2.0*20)**-2, (3.0*20)**-2,
                            (0.1*20)**-2, (30*20)**-2, 2.0**-2, 3.0**-2, 3.0**-2, 2.0**-2, 0.1**-2], device=device),
        'ic': torch.tensor([0.0] * 6, device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }
all_normalized_weights = {
        'pde': torch.tensor(np.array([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2,
                                      30.0**-2, 0.01**-2])*10, device=device),
        'bc': torch.tensor([3.0**-2, 2.0**-2, 0.1**-2,(30.0)**-2, (3.0*20)**-2, (2.0*20)**-2, (3.0*20)**-2,
                            (0.1*20)**-2, (30*20)**-2, 2.0**-2, 3.0**-2, 3.0**-2, 2.0**-2, 0.1**-2], device=device),
        'ic': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2, 0.01**-2], device=device),
        'sparse': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2, 0.01**-2], device=device)
    }

bc_super_ic_norm_weights = {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([3.0**-2, 2.0**-2, 0.1**-2,(30.0)**-2, (3.0*20)**-2, (2.0*20)**-2, (3.0*20)**-2,
                            (0.1*20)**-2, (30*20)**-2, 2.0**-2, 3.0**-2,
                            3.0**-2, 2.0**-2, 0.1**-2]*100, device=device),
        'ic': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2, 0.01**-2], device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }

bc_ic_super_all_norm_weights = {
        'pde': torch.tensor(np.array([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2,
                                      30.0**-2, 0.01**-2])*10, device=device),
        'bc': torch.tensor([3.0**-2, 2.0**-2, 0.1**-2,(30.0)**-2, (3.0*20)**-2, (2.0*20)**-2, (3.0*20)**-2,
                            (0.1*20)**-2, (30*20)**-2, 2.0**-2, 3.0**-2,
                            3.0**-2, 2.0**-2, 0.1**-2]*100, device=device),
        'ic': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2,
                            0.01**-2]*100, device=device),
        'sparse': torch.tensor([3.0**-2, 2.0**-2, 3.0**-2, 0.1**-2, 30.0**-2,
                                0.01**-2], device=device),
    }

def update_weights(model, inputs, boundary_conditions, initial_conditions,
                   sparse_data, weights, writer, epoch):
    x, y, t, Re, theta = inputs
    # Reset gradients for the model
    model.zero_grad()

    # Calculate loss components using the loss function
    _, loss_components = loss(model, inputs, boundary_conditions,
                              initial_conditions, sparse_data, weights, writer,
                              epoch)

    gradients = {}
    # Calculate gradients for each loss component type
    for key, losses in zip(['pde', 'bc', 'ic', 'sparse'], loss_components):
        grads = []
        for loss_ in losses:
            # Compute gradients and collect them
            loss_.backward(retain_graph=True)
            grads.append(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])))
            model.zero_grad()
        gradients[key] = grads

    # Determine the maximum value in the weights
    max_weight_value = max(weight.max().item() for weight in weights.values())

    N_gain = 10
    # Set the clamping threshold to N_gain times the maximum weight value
    clamp_threshold = max(N_gain * max_weight_value, 1000.0)
    # Calculate the total norm, skipping components with zero weights
    total_norm = sum([grad for key, grads in gradients.items() for grad, weight in zip(grads, weights[key]) if weight != 0])

    new_weights = {}

    # Update weights based on gradients, skipping zero weights and clamping if necessary
    for key, grads in gradients.items():
        weight_update = []
        for grad, weight in zip(grads, weights[key]):
            if weight != 0:
                # Compute the updated weight
                updated_weight = total_norm / grad
                weight_update.append(updated_weight)
            else:
                # Keep the weight as is if it is zero
                weight_update.append(weight)

        # Apply the scaling factor to the updated weights
        max_updated_weight = max(weight_update)
        if max_updated_weight > 10 * clamp_threshold: # introduced 10 not to shrink weights too much
            scaling_factor = 10 * clamp_threshold / max_updated_weight
            weight_update = [w * scaling_factor for w in weight_update]

        # Ensure weights do not go lower than 1.0, except for those that are zero
        weight_update = [w if weight == 0 else max(w, 1.0) for w, weight in zip(weight_update, weights[key])]
        weight_update = [min(w, 1000.0) for w in weight_update]

        new_weights[key] = torch.tensor(weight_update, device=device)

    # Example of using the formatted print function
    formatted_print("Updated weights:", new_weights)

    return new_weights

def bc_calc_loss(model, boundary_conditions, criterion):
    bc_losses = []
    for bc in boundary_conditions:
        inputs, conditions = bc
        x_b, y_b, t_b, Re_b, theta_b = inputs
        x_b.requires_grad_(True)
        y_b.requires_grad_(True)

        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([x_b, y_b, t_b, Re_b, theta_b], dim=1))

        for variable, condition in conditions.items():
            if condition['type'] == 'Dirichlet':
                value = condition['value']
                if variable == 'u':
                    bc_losses.append(criterion(u_pred, value))
                elif variable == 'v':
                    bc_losses.append(criterion(v_pred, value))
                elif variable == 'p':
                    bc_losses.append(criterion(p_pred, value))
                elif variable == 'k':
                    bc_losses.append(criterion(k_pred, value))
                elif variable == 'omega':
                    bc_losses.append(criterion(omega_pred, value))

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
    def __init__(self, data_array, num_intervals=100):
        self.data = torch.tensor(data_array, dtype=torch.float32).to(device)
        self.num_intervals = num_intervals
        self.interval_size = len(data_array) // num_intervals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_data = self.data[idx]
        x_sparse = batch_data[0].unsqueeze(0)
        y_sparse = batch_data[1].unsqueeze(0)
        t_sparse = batch_data[2].unsqueeze(0)
        Re_sparse = batch_data[3].unsqueeze(0)
        theta_sparse = batch_data[4].unsqueeze(0)

        u_sparse = batch_data[5].unsqueeze(0)
        v_sparse = batch_data[6].unsqueeze(0)
        p_sparse = batch_data[7].unsqueeze(0)
        k_sparse = batch_data[8].unsqueeze(0)
        omega_sparse = batch_data[9].unsqueeze(0)
        c_sparse = batch_data[10].unsqueeze(0)

        inputs = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse)
        outputs = (u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse)

        return inputs, outputs

    def get_batch(self, interval, batch_size):
        start_idx = (interval - 1) * self.interval_size
        end_idx = interval * self.interval_size
        selected_indices = torch.randint(start_idx, end_idx, (batch_size,), device=device)

        batch_data = self.data[selected_indices]
        x_sparse = batch_data[:, 0].unsqueeze(1)
        y_sparse = batch_data[:, 1].unsqueeze(1)
        t_sparse = batch_data[:, 2].unsqueeze(1)
        Re_sparse = batch_data[:, 3].unsqueeze(1)
        theta_sparse = batch_data[:, 4].unsqueeze(1)

        u_sparse = batch_data[:, 5].unsqueeze(1)
        v_sparse = batch_data[:, 6].unsqueeze(1)
        p_sparse = batch_data[:, 7].unsqueeze(1)
        k_sparse = batch_data[:, 8].unsqueeze(1)
        omega_sparse = batch_data[:, 9].unsqueeze(1)
        c_sparse = batch_data[:, 10].unsqueeze(1)

        inputs = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse)
        outputs = (u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse)

        return inputs, outputs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()

    input_min = [-2.5, -2.5, 1, 30e6, 0]
    input_max = [7.5, 2.5, 100, 70e6, 2 * np.pi]
    output_min = [-1, -1, -2, 0, 0, 0]
    output_max = [2, 1, 1, 0.1, 30, 0.01]

    model = PINN(input_min, input_max, output_min, output_max).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=optim_config.learning_rate,
        betas=(optim_config.beta1, optim_config.beta2),
        eps=optim_config.eps,
        weight_decay=1e-4  # Example value for weight decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_config.decay_steps, gamma=optim_config.decay_rate)

    # Load preprocessed data
    data_array = np.load("data/preprocessed_data.npy")
    dataset = CustomDataset(data_array)
    weights = all_ones_weights

    run_schedule = [
        (10000, all_normalized_weights),
    ]

    writer = SummaryWriter(log_dir='runs_/c19_lr5_gw') # TensorBoard

    # Initialize temporal weights
    '''
    temporal_weights_pde = torch.ones(100, device=device, requires_grad=True)
    temporal_weights_sparse = torch.ones(100, device=device, requires_grad=True)
    temporal_weights_bc = torch.ones(100, device=device, requires_grad=True)
    '''

    tot_epoch = 0
    for epochs, initial_weights in run_schedule:
        if initial_weights is not None:
            weights = initial_weights

        for epoch in range(epochs):
            total_loss = 0

            inputs_0, outputs_0 = dataset.get_batch(1, batch_size) # interval = 1
            x_0, y_0, t_0, Re_0, theta_0 = inputs_0
            u_0, v_0, p_0, k_0, omega_0, c_0 = outputs_0

            u_0_pred, v_0_pred, p_0_pred, k_0_pred, omega_0_pred, c_0_pred = \
                model(torch.cat([x_0, y_0, t_0, Re_0, theta_0], dim=1))
            ic_losses = [
                criterion(u_0_pred, u_0),
                criterion(v_0_pred, v_0),
                criterion(p_0_pred, p_0),
                criterion(k_0_pred, k_0),
                criterion(omega_0_pred, omega_0),
                criterion(c_0_pred, c_0)
            ]

            for interval in range(1, 101):
                interval_loss = 0
                batch_size = 32

                inputs, outputs = dataset.get_batch(interval, batch_size)
                x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse = inputs
                u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = outputs

                boundary_conditions = generate_boundary_conditions(batch_size, interval):
                bc_losses = bc_calc_loss(model, boundary_conditions, criterion)

                pde_inputs = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse)
                continuity_residual, x_momentum_residual, y_momentum_residual,\
                  k_residual, omega_residual, c_residual = pde_residuals(model, *pde_inputs)

                pde_losses = [
                    criterion(continuity_residual, torch.zeros_like(continuity_residual)),
                    criterion(x_momentum_residual, torch.zeros_like(x_momentum_residual)),
                    criterion(y_momentum_residual, torch.zeros_like(y_momentum_residual)),
                    criterion(k_residual, torch.zeros_like(k_residual)),
                    criterion(omega_residual, torch.zeros_like(omega_residual)),
                    criterion(c_residual, torch.zeros_like(c_residual))
                ]

                sparse_data = (x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse,\
                  u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse)

                u_sparse_pred, v_sparse_pred, p_sparse_pred, k_sparse_pred,\
                  omega_sparse_pred, c_sparse_pred = model(torch.cat(\
                    [x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse], dim=1))

                sparse_losses = [
                    criterion(u_sparse_pred, u_sparse),
                    criterion(v_sparse_pred, v_sparse),
                    criterion(p_sparse_pred, p_sparse),
                    criterion(k_sparse_pred, k_sparse),
                    criterion(omega_sparse_pred, omega_sparse),
                    criterion(c_sparse_pred, c_sparse)
                ]

                interval_loss += sum(pde_losses) + sum(bc_losses) + sum(ic_losses) + sum(sparse_losses)
                total_loss += interval_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % N_loss_print == 0:
                formatted_print(f'Epoch {epoch}, Loss: {total_loss.item()}')

            if epoch % N_save_model == 0:
                save_model(model, 'c18_model.pth')

            if epoch % N_weight_update == 0 and epoch != 0:
                weights = update_weights(model, pde_inputs, boundary_conditions, initial_conditions, sparse_data, weights, writer, tot_epoch)

            scheduler.step()
            tot_epoch += 1

if __name__ == "__main__":
    main()

#print(f"main: Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}",
# f"MB Reserved Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
