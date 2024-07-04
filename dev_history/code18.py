import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from utils import calculate_range_and_divergence, check_for_nan_and_inf
from utils import formatted_print, plot_fields, check_tensor_stats
from ml_collections import ConfigDict
from torch.profiler import profile, record_function, ProfilerActivity

print('run 100223')
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

# Set the seed
set_seed(10004)

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

N_bc = 200
N_ic = 1000
N_pde = 2000
N_sparse = 2000

N_loss_print = 10
N_save_model = 1000
N_weight_update = 10
N_plot_fields = 100

# Load dataset
def get_dataset():
    file = "data/unsteady_smh.npy"
    data = np.load(file, allow_pickle=True).item()
    u_ref = np.array(data["u"].astype(float))
    v_ref = np.array(data["v"].astype(float))
    p_ref = np.array(data["p"].astype(float))
    k_ref = np.array(data["k"].astype(float))
    omega_ref = np.array(data["omega"].astype(float))
    c_ref = np.array(data["c"].astype(float))
    coords = np.array(data["coords"])
    Re = np.array(data["Re"])

    return u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re

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
        return u, v, p, k, omega, c

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
    U_star = 9.0
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

def train_step(model, optimizer, pde_inputs, boundary_conditions,
               initial_conditions, sparse_data, weights, writer, epoch):
    optimizer.zero_grad()
    total_loss, _ = loss(model, pde_inputs, boundary_conditions,
                         initial_conditions, sparse_data, weights, writer,
                         epoch)

    total_loss.backward()

    # Calculate the gradient norm before clipping
    total_norm_before = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optim_config.clip_norm)

    # Calculate the gradient norm after clipping
    total_norm_after = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None))

    # Print or log the gradient norms
#   print(f'Grad Norm Before Clipping: {total_norm_before:.4f}, '
#         f'Grad Norm After Clipping: {total_norm_after:.4f}')

    optimizer.step()
    return total_loss.item()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))

def get_nondim_dataset():
    u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re = get_dataset()

    U_star = 9.0 # for sim. 0

    T_star = L_star / U_star

    coords = coords / L_star
    u_ref = u_ref / U_star
    v_ref = v_ref / U_star
    p_ref = p_ref / U_star**2
    k_ref = k_ref / U_star**2
    omega_ref = omega_ref * L_star / U_star

    # clip negative & extreme values
    u_ref = np.clip(u_ref, 0.0, None)
    v_ref = np.clip(v_ref, 0.0, None)
    p_ref = np.clip(p_ref, 0.0, None)
    k_ref = np.clip(k_ref, 0.0, None)
    omega_ref = np.clip(omega_ref, 0.0, 30)
    c_ref = np.clip(c_ref, 0.0, 0.01)

    return u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re, U_star, L_star

def generate_initial_conditions(device):
    u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re, U_star, L_star = get_nondim_dataset()

    calculate_range_and_divergence(u_ref,'u_ref')
    calculate_range_and_divergence(v_ref,'v_ref')
    calculate_range_and_divergence(p_ref,'p_ref')
    calculate_range_and_divergence(k_ref,'k_ref')
    calculate_range_and_divergence(omega_ref,'omega_ref')
    calculate_range_and_divergence(c_ref,'c_ref')
    calculate_range_and_divergence(coords,'coords')

    x_0 = torch.tensor(coords[:, 0], dtype=torch.float32).unsqueeze(1).to(device)
    y_0 = torch.tensor(coords[:, 1], dtype=torch.float32).unsqueeze(1).to(device)
    t_0 = torch.full_like(x_0, U_star/L_star).to(device)
    Re_0 = torch.full_like(x_0, Re.item()).to(device)
    theta_0 = torch.zeros_like(x_0).to(device)
    u_0 = torch.tensor(u_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    v_0 = torch.tensor(v_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    p_0 = torch.tensor(p_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    k_0 = torch.tensor(k_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    omega_0 = torch.tensor(omega_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    c_0 = torch.tensor(c_ref[0], dtype=torch.float32).unsqueeze(1).to(device)

    return x_0, y_0, t_0, Re_0, theta_0, u_0, v_0, p_0, k_0, omega_0, c_0

def generate_boundary_conditions(device):
    Re_medium = 1.225 * 9.0 * 80.0 / 1.7894e-5
    k_in_value = 3/2*(0.05*9.0)**2/9.0**2
    omega_in_value = 30.0

    N_each_bk = 20000
    N_bc_in = N_each_bk
    N_bc_sym = N_each_bk
    N_bc_out = N_each_bk
    N_bc_wall = N_each_bk

    x_in = torch.full((N_bc_in, 1), -200.0/L_star).to(device)
    y_in = torch.linspace(-200/L_star, 200/L_star, N_bc_in).view(-1, 1).to(device)
    t_in = torch.randint(1, 101, x_in.shape, device=device).float()
    Re_in = torch.full((N_bc_in, 1), Re_medium).to(device)
    theta_in = torch.zeros_like(x_in).to(device)
    u_in = torch.full((N_bc_in, 1), 1.0).to(device)
    v_in = torch.zeros_like(u_in).to(device)
    k_in = torch.full((N_bc_in, 1), k_in_value).to(device)
    omega_in = torch.full((N_bc_in, 1), omega_in_value).to(device)

    x_sym = torch.linspace(-200/L_star, 600/L_star, N_bc_sym).view(-1, 1).to(device)
    y_sym = (torch.where(torch.randint(0, 2, (N_bc_sym, 1), device=device) == 0, -200.0, 200.0) / L_star).to(device)
    t_sym = torch.randint(1, 101, x_sym.shape, device=device).float()
    Re_sym = torch.full((N_bc_sym, 1), Re_medium).to(device)
    theta_sym = torch.zeros_like(x_sym).to(device)
    u_sym = torch.zeros_like(x_sym).to(device)
    v_sym = torch.zeros_like(x_sym).to(device)
    p_sym = torch.zeros_like(x_sym).to(device)
    k_sym = torch.zeros_like(x_sym).to(device)
    omega_sym = torch.zeros_like(x_sym).to(device)

    x_out = torch.full((N_bc_out, 1), 600.0/L_star).to(device)
    y_out = torch.linspace(-200/L_star, 200/L_star, N_bc_out).view(-1, 1).to(device)
    t_out = torch.randint(1, 101, x_out.shape, device=device).float()
    Re_out = torch.full((N_bc_out, 1), Re_medium).to(device)
    theta_out = torch.zeros_like(x_out).to(device)
    p_out = torch.zeros((N_bc_out, 1)).to(device)

    theta_rand = torch.linspace(0, 2 * np.pi, N_bc_wall).to(device)
    x_wall = (40/L_star * torch.cos(theta_rand)).view(-1, 1).to(device)
    y_wall = (40/L_star * torch.sin(theta_rand)).view(-1, 1).to(device)
    t_wall = torch.randint(1, 101, x_wall.shape, device=device).float()
    Re_wall = torch.full((N_bc_wall, 1), Re_medium).to(device)
    theta_wall = torch.zeros_like(x_wall).to(device)
    u_wall = torch.zeros_like(x_wall).to(device)
    v_wall = torch.zeros_like(x_wall).to(device)
    k_wall = torch.zeros_like(x_wall).to(device)

    boundary_conditions = [
        (x_in, y_in, t_in, Re_in, theta_in, {
            'u': {'type': 'Dirichlet', 'value': u_in},
            'v': {'type': 'Dirichlet', 'value': v_in},
            'k': {'type': 'Dirichlet', 'value': k_in},
            'omega': {'type': 'Dirichlet', 'value': omega_in},
        }),
        (x_sym, y_sym, t_sym, Re_sym, theta_sym, {
            'u': {'type': 'Neumann', 'dir_deriv': 'y', 'value': u_sym},
            'v': {'type': 'Neumann', 'dir_deriv': 'y', 'value': v_sym},
            'p': {'type': 'Neumann', 'dir_deriv': 'y', 'value': p_sym},
            'k': {'type': 'Neumann', 'dir_deriv': 'y', 'value': k_sym},
            'omega': {'type': 'Neumann', 'dir_deriv': 'y', 'value': omega_sym}
        }),
        (x_sym, y_sym, t_sym, Re_sym, theta_sym, {
            'v': {'type': 'Dirichlet', 'value': v_sym}
        }),
        (x_out, y_out, t_out, Re_out, theta_out, {
            'p': {'type': 'Dirichlet', 'value': p_out}
        }),
        (x_wall, y_wall, t_wall, Re_wall, theta_wall, {
            'u': {'type': 'Dirichlet', 'value': u_wall},
            'v': {'type': 'Dirichlet', 'value': v_wall},
            'k': {'type': 'Dirichlet', 'value': k_wall},
        })
    ]
    return boundary_conditions

def generate_sparse_data(device):
    u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re, U_star, L_star = get_nondim_dataset()
    x_sparse = torch.tensor(np.tile(coords[:,0], len(u_ref)), dtype=torch.float32).unsqueeze(1).to(device)
    y_sparse = torch.tensor(np.tile(coords[:,1], len(u_ref)), dtype=torch.float32).unsqueeze(1).to(device)
    t_sparse = torch.arange(1.0, u_ref.shape[0] + 1.0).repeat_interleave(u_ref.shape[1]).unsqueeze(1).to(device)*(U_star/L_star)
    Re_sparse = torch.full_like(x_sparse, Re.item()).to(device)
    theta_sparse = torch.zeros_like(x_sparse).to(device)
    u_sparse = torch.tensor(u_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    v_sparse = torch.tensor(v_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    p_sparse = torch.tensor(p_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    k_sparse = torch.tensor(k_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    omega_sparse = torch.tensor(omega_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    c_sparse = torch.tensor(c_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    return x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse

def loss(model, pde_inputs, boundary_conditions, initial_conditions,
         sparse_data, weights, writer, epoch):
    continuity_residual, x_momentum_residual, y_momentum_residual, k_residual,\
      omega_residual, c_residual = pde_residuals(model, *pde_inputs)

    pde_losses = [
        torch.mean(continuity_residual ** 2),
        torch.mean(x_momentum_residual ** 2),
        torch.mean(y_momentum_residual ** 2),
        torch.mean(k_residual ** 2),
        torch.mean(omega_residual ** 2),
        torch.mean(c_residual ** 2)
    ]

    loss_pde = sum([weights['pde'][i] * pde_losses[i] for i in range(len(pde_losses))])

    bc_losses = []
    for bc in boundary_conditions:
        x_b, y_b, t_b, Re_b, theta_b, conditions = bc
        x_b.requires_grad_(True)
        y_b.requires_grad_(True)
        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([x_b, y_b, t_b, Re_b, theta_b], dim=1))

        for variable, condition in conditions.items():
            if condition['type'] == 'Dirichlet':
                value = condition['value']
                if variable == 'u':
                    bc_losses.append(torch.mean((u_pred - value) ** 2))
                elif variable == 'v':
                    bc_losses.append(torch.mean((v_pred - value) ** 2))
                elif variable == 'p':
                    bc_losses.append(torch.mean((p_pred - value) ** 2))
                elif variable == 'k':
                    bc_losses.append(torch.mean((k_pred - value) ** 2))
                elif variable == 'omega':
                    bc_losses.append(torch.mean((omega_pred - value) ** 2))

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

                bc_losses.append(torch.mean((deriv - value) ** 2))
    loss_bc = sum([weights['bc'][i] * bc_losses[i] for i in range(len(bc_losses))])

    # Calculate the loss for the initial conditions
    x_0, y_0, t_0, Re_0, theta_0, u_0, v_0, p_0, k_0, omega_0, c_0 = initial_conditions
    u_0_pred, v_0_pred, p_0_pred, k_0_pred, omega_0_pred, c_0_pred = \
    model(torch.cat([x_0, y_0, t_0, Re_0, theta_0], dim=1))
    ic_losses = [
        torch.mean((u_0_pred - u_0) ** 2),
        torch.mean((v_0_pred - v_0) ** 2),
        torch.mean((p_0_pred - p_0) ** 2),
        torch.mean((k_0_pred - k_0) ** 2),
        torch.mean((omega_0_pred - omega_0) ** 2),
        torch.mean((c_0_pred - c_0) ** 2)
    ]
    loss_ic = sum([weights['ic'][i] * ic_losses[i] for i in range(len(ic_losses))])

    x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = sparse_data
    u_sparse_pred, v_sparse_pred, p_sparse_pred, k_sparse_pred, omega_sparse_pred, c_sparse_pred = model(torch.cat([x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse], dim=1))
    sparse_losses = [
        torch.mean((u_sparse_pred - u_sparse) ** 2),
        torch.mean((v_sparse_pred - v_sparse) ** 2),
        torch.mean((p_sparse_pred - p_sparse) ** 2),
        torch.mean((k_sparse_pred - k_sparse) ** 2),
        torch.mean((omega_sparse_pred - omega_sparse) ** 2),
        torch.mean((c_sparse_pred - c_sparse) ** 2)
    ]
    loss_sparse = sum([weights['sparse'][i] * sparse_losses[i] for i in range(len(sparse_losses))])

    # Apply temporal weights to sparse data losses
    # Apply temporal weights to pde losses

    total_loss = loss_pde + loss_bc + loss_ic + loss_sparse

# ---------------------------- Tensorboard -----------------------------------
    bc_names = [
        'inlet_u', 'inlet_v', 'inlet_k', 'inlet_omega',
        'symmetry_u', 'symmetry_v', 'symmetry_p', 'symmetry_k', 'symmetry_omega',
        'symmetry_v_dirichlet', 'outlet_p', 'wall_u', 'wall_v', 'wall_k' ]
    ic_names = ['u', 'v', 'p', 'k', 'omega', 'c']
    sparse_names = ['u', 'v', 'p', 'k', 'omega', 'c']
    pde_names = ['continuity', 'x_momentum', 'y_momentum', 'k_transport',
                      'omega_transport', 'convection_diffusion']

    writer.add_scalar('_total_loss', total_loss, epoch)

    # Log losses
    writer.add_scalar('bc/_total', loss_bc, epoch)
    for i, bc_loss in enumerate(bc_losses):
        writer.add_scalar(f'bc/{bc_names[i]}', weights['bc'][i] * bc_loss, epoch)

    writer.add_scalar('ic/_total', loss_ic, epoch)
    for i, ic_loss in enumerate(ic_losses):
        writer.add_scalar(f'ic/{ic_names[i]}', weights['ic'][i] * ic_loss, epoch)

    writer.add_scalar('sparse/_total', loss_sparse, epoch)
    for i, sparse_loss in enumerate(sparse_losses):

        writer.add_scalar(f'sparse/{sparse_names[i]}', weights['sparse'][i] * sparse_loss, epoch)
    writer.add_scalar('pde/_total', loss_pde, epoch)
    for i, pde_loss in enumerate(pde_losses):
        writer.add_scalar(f'pde/{pde_names[i]}', weights['pde'][i] * pde_loss, epoch)

    # Log weights
    for i, bc_loss in enumerate(bc_losses):
        writer.add_scalar(f'w_bc/{bc_names[i]}', weights['bc'][i], epoch)

    for i, ic_loss in enumerate(ic_losses):
        writer.add_scalar(f'w_ic/{ic_names[i]}', weights['ic'][i], epoch)

    for i, sparse_loss in enumerate(sparse_losses):
        writer.add_scalar(f'w_sparse/{sparse_names[i]}', weights['sparse'][i], epoch)

    for i, pde_loss in enumerate(pde_losses):
        writer.add_scalar(f'w_pde/{pde_names[i]}', weights['pde'][i], epoch)

    # Log plots
    if epoch % N_plot_fields == 0:
        inputs = x_0, y_0, t_0, Re_0, theta_0
        time = 1.0
        fig = plot_fields(time, model, *inputs, f'code17h_{epoch}') # provide U_star for dimensional plot
        writer.add_figure('Predicted Fields', fig, epoch)

    return total_loss, [pde_losses, bc_losses, ic_losses, sparse_losses]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    boundary_conditions = generate_boundary_conditions(device)
    initial_conditions = generate_initial_conditions(device)
    sparse_data = generate_sparse_data(device)
    x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse,\
      p_sparse, k_sparse, omega_sparse, c_sparse = sparse_data
    weights = all_ones_weights

    '''
    run_schedule = [
        (1000, bc_normalized_weights),
        (1000, bc_super_ic_norm_weights),
        (1000, bc_ic_super_all_norm_weights),
    ]
    '''
    run_schedule = [
        (10000, all_normalized_weights),
    ]

    writer = SummaryWriter(log_dir='runs/c18_r1_lr_e-5_weight') # TensorBoard

    # Initialize temporal weights
    temporal_weights_pde = torch.ones(100, device=device, requires_grad=True)
    temporal_weights_sparse = torch.ones(100, device=device, requires_grad=True)

    tot_epoch = 0
    for epochs, initial_weights in run_schedule:
        if initial_weights is not None:
            weights = initial_weights

        for epoch in range(epochs):
            pde_indices = torch.randperm(len(x_sparse))[:N_pde]

            mu_rho_u2 = mu/rho/L_star**2
            t_pde = (mu_rho_u2 * Re_sparse[pde_indices].squeeze() +
                     99 * mu_rho_u2 * Re_sparse[pde_indices].squeeze() *
                     torch.rand(N_pde, device=device)).unsqueeze(1)

            pde_data_subset = [
                d[pde_indices].unsqueeze(1) if len(d.shape) == 1 else d[pde_indices]
                if d is not t_pde else d for d in [x_sparse, y_sparse, t_pde, Re_sparse, theta_sparse]
            ]

            bc_conditions_subset = []
            for bc in boundary_conditions:
                x_bc, y_bc, t_bc, Re_bc, theta_bc, conditions = bc
                bc_indices = torch.randperm(len(x_bc))[:N_bc]
                subset_conditions = {}
                for var in conditions:
                    subset_conditions[var] = {
                        'type': conditions[var]['type'],
                        'value': conditions[var]['value'][bc_indices]
                    }
                    if 'dir_deriv' in conditions[var]:
                        subset_conditions[var]['dir_deriv'] = conditions[var]['dir_deriv']
                bc_subset = (x_bc[bc_indices], y_bc[bc_indices], t_bc[bc_indices], Re_bc[bc_indices], theta_bc[bc_indices], subset_conditions)
                bc_conditions_subset.append(bc_subset)

            sparse_indices = torch.randperm(len(x_sparse))[:N_sparse]
            sparse_data_subset = [d[sparse_indices].unsqueeze(1) if len(d.shape) == 1 else d[sparse_indices] for d in sparse_data]

            ic_indices = torch.randperm(len(initial_conditions[0]))[:N_ic]
            initial_conditions_subset = [ic[ic_indices] for ic in initial_conditions]

            loss_value = train_step(model, optimizer, pde_data_subset,
                                    bc_conditions_subset,
                                    initial_conditions_subset,
                                    sparse_data_subset, weights, writer, tot_epoch)

            if epoch % N_loss_print == 0:
                formatted_print(f'Epoch {epoch}, Loss: {loss_value}')

            if epoch % N_save_model == 0:
                save_model(model, 'c18_model.pth')

            if epoch % N_weight_update == 0 and epoch != 0:
                weights = update_weights(model, pde_data_subset,
                                         bc_conditions_subset,
                                         initial_conditions_subset,
                                         sparse_data_subset, weights, writer
                                         , tot_epoch)

            scheduler.step()
            tot_epoch = tot_epoch + 1 #TensorBoard

if __name__ == "__main__":
    main()
