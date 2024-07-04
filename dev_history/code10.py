import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch.nn.functional as F
from utils import calculate_range_and_divergence
from utils import check_for_nan_and_inf
from utils import check_tensor_stats
#from monitor import ActivationMonitor
from monitor import forward_hook
from monitor import register_hooks

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

# Define global constants for SST k-omega model
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
#U_star = 9.0  # [m/s] velocity
#L_star = 80.0  # [m] diameter
#Re_medium = rho * U_star * L_star / mu

# Define a function to check if points are inside the circle
def inside_circle(x, y, center_x=0, center_y=0, radius=40):
    return (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2

def get_dataset():
    file = "data/unsteady_nmh.npy"
    data = np.load(file, allow_pickle=True).item()
    u_ref = np.array(data["u"].astype(float))
    v_ref = np.array(data["v"].astype(float))
    p_ref = np.array(data["p"].astype(float))
    k_ref = np.array(data["k"].astype(float))
    omega_ref = np.array(data["omega"].astype(float))
    c_ref = np.array(data["c"].astype(float))
    #t = np.array(data["t"])
    coords = np.array(data["coords"])
    Re = np.array(data["Re"])

    return (
        u_ref,
        v_ref,
        p_ref,
        k_ref,
        omega_ref,
        c_ref,
        coords,
        Re
    )

class PINN(nn.Module):
    def __init__(self, input_min, input_max, output_min, output_max):
        super(PINN, self).__init__()
        N1 = 128  # Number of neurons
        self.normalization = NormalizationLayer(input_min, input_max)
        self.fc1 = nn.Linear(5, N1)
        self.fc2 = nn.Linear(N1, N1)
        self.fc3 = nn.Linear(N1, N1)
        self.fc4 = nn.Linear(N1, N1)
        self.fc_out = nn.Linear(N1, 6)  # Combine outputs into a single layer
        self.denormalization = DenormalizationLayer(output_min, output_max)
        self._initialize_weights()

    def forward(self, x):
        x = self.normalization(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
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

    '''
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization for weights
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    '''
    '''
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization for weights
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)  # Uniform initialization for biases
    '''

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


def smooth_maximum(a, b, alpha=10):
    b = b.expand_as(a)  # Ensure b has the same shape as a
    return torch.logsumexp(torch.stack([a, b], dim=0) * alpha, dim=0) / alpha

def smooth_minimum(a, b, alpha=10):
    b = b.expand_as(a)  # Ensure b has the same shape as a
    return -torch.logsumexp(torch.stack([-a, -b], dim=0) * alpha, dim=0) / alpha

def smooth_conditional(cond, true_val, false_val, alpha=10):
    true_val = true_val.expand_as(cond)  # Ensure true_val has the same shape as cond
    false_val = false_val.expand_as(cond)  # Ensure false_val has the same shape as cond
    return cond.sigmoid() * true_val + (1 - cond.sigmoid()) * false_val

def safe_sqrt(tensor, epsilon=1e-8):
    return torch.sqrt(tensor + epsilon)

def ensure_positive(tensor, epsilon=1e-10):
    print("ensure_positive")
    return torch.nn.functional.relu(tensor) + epsilon

# Define the PDE residuals for the SST k-omega model and convection-diffusion equation
def pde_residuals(model, x, y, t, Re, theta):
    L_star = 80.0
    U_star = 9.0
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
#    Re.requires_grad_(True)
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

# Clamping mu_t to avoid extreme values
    phi_11 = torch.clamp(phi_11, min=-1e10, max=1e6)
    phi_12 = torch.clamp(phi_12, min=-1e10, max=1e6)
    phi_13 = torch.clamp(phi_13, min=-1e10, max=1e6)
    phi_1 = torch.clamp(phi_1, min=-1e10, max=1e6)
    phi_21 = torch.clamp(phi_21, min=-1e10, max=1e6)
    phi_22 = torch.clamp(phi_22, min=-1e10, max=1e6)
    phi_2 = torch.clamp(phi_2, min=-1e10, max=1e6)

    dummy_1 = torch.autograd.grad(safe_sqrt(k), y,
           grad_outputs=torch.ones_like(k), create_graph=True)[0]


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
# Clamping mu_t to avoid extreme values
    mu_t = torch.clamp(mu_t, min=1e-10, max=1e6)

    G_k = mu_t * S ** 2
    Y_k = beta_star * k * omega
    G_k_tilde = smooth_minimum(G_k, 10 * beta_star * k * omega)

    G_omega = alpha / mu_t * G_k_tilde
    Y_omega = beta * omega ** 2
    D_omega = 2 * (1 - F1) * (sigma_omega2 / omega) * (k_x * omega_x + k_y * omega_y)

    continuity_residual = u_x + v_y
    x_mom_x = (1/Re + mu_t) * (4/3 * u_x - 2/3 * v_y)
# Zero out values in x_mom_x that are smaller than the threshold x_mom_x = torch.where(torch.abs(x_mom_x) < 1e-6, torch.tensor(0.0, device=x_mom_x.device), x_mom_x)

    x_mom_y = (1/Re + mu_t) * (v_x + u_y)


    x_mom_gradx = torch.autograd.grad(x_mom_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_mom_grady = torch.autograd.grad(x_mom_y, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_momentum_residual = u_t + u * u_x + v * u_y + p_x - x_mom_gradx - x_mom_grady
    y_mom_grady = torch.autograd.grad((1/Re + mu_t) * (4/3 * v_y - 2/3 * u_x), y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_mom_gradx = torch.autograd.grad((1/Re + mu_t) * (v_x + u_y), x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_momentum_residual = v_t + u * v_x + v * v_y + p_y - y_mom_grady - y_mom_gradx

    k_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_x, x, grad_outputs=torch.ones_like(k_x), create_graph=True)[0]
    k_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_y, y, grad_outputs=torch.ones_like(k_y), create_graph=True)[0]

# Clamping intermediate transport terms
#    k_transport_term1 = torch.clamp(k_transport_term1, min=-10.0, max=10.0)
#   k_transport_term2 = torch.clamp(k_transport_term2, min=-10.0, max=10.0)
#   G_k = torch.clamp(G_k, min=-10.0, max=10.0)
#   Y_k = torch.clamp(Y_k, min=-10.0, max=10.0)

    k_residual = k_t + u * k_x + v * k_y - k_transport_term1 - k_transport_term2 - G_k + Y_k

    omega_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]

    # Clamping intermediate transport terms
    omega_transport_term1 = torch.clamp(omega_transport_term1, min=-10.0, max=10.0)
    omega_transport_term2 = torch.clamp(omega_transport_term2, min=-10.0, max=10.0)
    G_omega = torch.clamp(G_omega, min=-10.0, max=10.0)
    Y_omega = torch.clamp(Y_omega, min=-10.0, max=10.0)
    D_omega = torch.clamp(D_omega, min=-10.0, max=10.0)


    omega_residual = omega_t + u * omega_x + v * omega_y - omega_transport_term1 - omega_transport_term2 - G_omega + Y_omega - D_omega
    c_residual = c_t + u * c_x + v * c_y - (1 / Re) * (c_x + c_y)  # Convection-diffusion equation
# ------------------------------------------------- debug --------------------
    pde_check_list = {
        'x':x,
        'y':y,
        't':t,
        'Re':Re,
        'theta':theta,
        'u': u,
        'v': v,
        'p': p,
        'k': k,
        'omega': omega,
        'c': c,
        'Re': Re,
        'theta': theta,
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
#   for key in pde_check_list:
#       check_for_nan_and_inf(pde_check_list[key], key)

#check_for_nan_and_inf(x_momentum_residual,'x_mom')
#    check_for_nan_and_inf(y_momentum_residual,'y_mom')
#   check_for_nan_and_inf(k_residual, 'k_trn')
#  check_for_nan_and_inf(omega_residual, 'omega_trn')
# check_for_nan_and_inf(c_residual, 'conv-diff')

    return continuity_residual, x_momentum_residual, y_momentum_residual, k_residual, omega_residual, c_residual

# Function to update weights
def update_weights(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data):
    model.zero_grad()

    # Compute gradients for each loss component
    _, loss_components = loss(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data, default_weights())

    # Compute gradients for each loss component
    gradients = {}
    for key, losses in zip(['pde', 'bc', 'ic', 'sparse'], loss_components):
        grads = []
        for loss_ in losses:
            loss_.backward(retain_graph=True)
            grads.append(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])))
            model.zero_grad()
        gradients[key] = grads

    # Compute global weights
    total_norm = sum([sum(grads) for grads in gradients.values()])
    weights = {key: torch.tensor([total_norm / grad for grad in grads], device=device) for key, grads in gradients.items()}

    return weights

# Function to return default weights
def default_weights():
    return {
        'pde': torch.tensor([0.0] * 6, device=device),
        'bc': torch.tensor([1.0] * 14, device=device),
        'ic': torch.tensor([0.0] * 6, device=device),
        'sparse': torch.tensor([0.0] * 6, device=device)
    }

# Define the training step
def train_step(model, optimizer, x, y, t, Re, theta, boundary_conditions\
               , initial_conditions, sparse_data, weights):

    optimizer.zero_grad()
    total_loss, _ = loss(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data, weights)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients here
    optimizer.step()
    return total_loss.item()

# Save model weights
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model weights
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))

def get_nondim_dataset():
    u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re = get_dataset()

    U_star = 9.0 # for sim. 0
    L_star = 80.0 # for sim. 0

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

    return (
        u_ref,
        v_ref,
        p_ref,
        k_ref,
        omega_ref,
        c_ref,
        coords,
        Re,
        U_star,
        L_star
    )

def generate_initial_conditions(device):
    u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, coords, Re, U_star, L_star = get_nondim_dataset()

# ------------------------------- debug --------------------------------------
    calculate_range_and_divergence(u_ref,'u_ref')
    calculate_range_and_divergence(v_ref,'v_ref')
    calculate_range_and_divergence(p_ref,'p_ref')
    calculate_range_and_divergence(k_ref,'k_ref')
    calculate_range_and_divergence(omega_ref,'omega_ref')
    calculate_range_and_divergence(c_ref,'c_ref')
    calculate_range_and_divergence(coords,'coords')

    x_0 = torch.tensor(coords[:, 0], dtype=torch.float32).unsqueeze(1).to(device)
    y_0 = torch.tensor(coords[:, 1], dtype=torch.float32).unsqueeze(1).to(device)
    t_0 = torch.full_like(x_0, U_star/L_star).to(device) # init t=1s, t* = u*/l*
    Re_0 = torch.full_like(x_0, Re.item()).to(device)
    theta_0 = torch.zeros_like(x_0).to(device)
    u_0 = torch.tensor(u_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    v_0 = torch.tensor(v_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    p_0 = torch.tensor(p_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    k_0 = torch.tensor(k_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    omega_0 = torch.tensor(omega_ref[0], dtype=torch.float32).unsqueeze(1).to(device)
    c_0 = torch.tensor(c_ref[0], dtype=torch.float32).unsqueeze(1).to(device)

    return x_0, y_0, t_0, Re_0, theta_0, u_0, v_0, p_0, k_0, omega_0, c_0

# Generate boundary conditions
def generate_boundary_conditions(device):
    # Velocity Inlet (Left Boundary: x = -200, -200 ≤ y ≤ 200)
    Re_medium = 1.225 * 9.0 * 80.0 / 1.7894e-5  # for now
    k_in_value = 3/2*(0.05*9.0)**2/9.0**2 # for now
    #omega_in_value = (3/2*(0.05*9.0)**2/9.0**2)/(10*1.7894e-5/1.225) # sst k-w
    omega_in_value = 30.0
    L_star = 80.0

    N_bc = 1002
    N_bc_in = N_bc
    N_bc_sym = N_bc
    N_bc_out = N_bc
    N_bc_wall = N_bc

    # Constant Velocity Inlet (Left Boundary: y = -200, -200 ≤ y ≤ 200)
    x_in = torch.full((N_bc_in, 1), -200.0/L_star).to(device)
    y_in = torch.linspace(-200/L_star, 200/L_star, N_bc_in).view(-1, 1).to(device)
    t_in = torch.randint(1, 101, x_in.shape, device=device).float()
    Re_in = torch.full((N_bc_in, 1), Re_medium).to(device) # for now
    theta_in = torch.zeros_like(x_in).to(device)
    u_in = torch.full((N_bc_in, 1), 1.0).to(device)
    v_in = torch.zeros_like(u_in).to(device)
    k_in = torch.full((N_bc_in, 1), k_in_value).to(device) # for now
    omega_in = torch.full((N_bc_in, 1), omega_in_value).to(device) # for now

    # Symmetry (Top and Bottom Boundaries: -200 ≤ x ≤ 600, y = ±200)
    x_sym = torch.linspace(-200/L_star, 600/L_star, N_bc_sym).view(-1, 1).to(device)
    y_sym = (torch.where(torch.randint(0, 2, (N_bc_sym, 1), device=device) == 0, -200.0, 200.0) / L_star).to(device)
    t_sym = torch.randint(1, 101, x_sym.shape, device=device).float()
    Re_sym = torch.full((N_bc_sym, 1), Re_medium).to(device)
    theta_sym = torch.zeros_like(x_sym).to(device)
    u_sym = torch.zeros_like(x_sym).to(device)
    v_sym = torch.zeros_like(x_sym).to(device) # for both neumann and dirichlet
    p_sym = torch.zeros_like(x_sym).to(device)
    k_sym = torch.zeros_like(x_sym).to(device)
    omega_sym = torch.zeros_like(x_sym).to(device)

    # Constant Pressure Outlet (Right Boundary: x = 600, -200 ≤ y ≤ 200)
    x_out = torch.full((N_bc_out, 1), 600.0/L_star).to(device)
    y_out = torch.linspace(-200/L_star, 200/L_star, N_bc_out).view(-1, 1).to(device)
    t_out = torch.randint(1, 101, x_out.shape, device=device).float()
    Re_out = torch.full((N_bc_out, 1), Re_medium).to(device)
    theta_out = torch.zeros_like(x_out).to(device)
    p_out = torch.zeros((N_bc_out, 1)).to(device)  # Dirichlet p = 0

    # Circular Wall (Radius 40, Centered at (0,0))
    theta_rand = torch.linspace(0, 2 * np.pi, N_bc_wall).to(device)
    x_wall = (40/L_star * torch.cos(theta_rand)).view(-1, 1).to(device)
    y_wall = (40/L_star * torch.sin(theta_rand)).view(-1, 1).to(device)
    t_wall = torch.randint(1, 101, x_wall.shape, device=device).float()
    Re_wall = torch.full((N_bc_wall, 1), Re_medium).to(device)
    theta_wall = torch.zeros_like(x_wall).to(device)
    u_wall = torch.zeros_like(x_wall).to(device) # u, v, k are zero
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
    t_sparse = torch.arange(1.0, u_ref.shape[0] + 1.0).\
        repeat_interleave(u_ref.shape[1]).unsqueeze(1).to(device)*(U_star/L_star)
    Re_sparse = torch.full_like(x_sparse, Re.item()).to(device)
    theta_sparse = torch.zeros_like(x_sparse).to(device)
    u_sparse = torch.tensor(u_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    v_sparse = torch.tensor(v_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    p_sparse = torch.tensor(p_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    k_sparse = torch.tensor(k_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    omega_sparse = torch.tensor(omega_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    c_sparse = torch.tensor(c_ref, dtype=torch.float32).reshape(-1, 1).to(device)

    return x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse\
        , v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse

def loss(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data, weights):
    # PDE residuals
    continuity_residual, x_momentum_residual, y_momentum_residual, k_residual, omega_residual, c_residual = pde_residuals(model, x, y, t, Re, theta)

    # Losses for PDE
    pde_losses = [
        torch.mean(continuity_residual ** 2),
        torch.mean(x_momentum_residual ** 2),
        torch.mean(y_momentum_residual ** 2),
        torch.mean(k_residual ** 2),
        torch.mean(omega_residual ** 2),
        torch.mean(c_residual ** 2)
    ]

    loss_pde = sum([weights['pde'][i] * pde_losses[i] for i in range(len(pde_losses))])

    # Boundary conditions loss
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

    # Initial conditions loss
    x_0, y_0, t_0, Re_0, theta_0, u_0, v_0, p_0, k_0, omega_0, c_0 = initial_conditions
    u_0_pred, v_0_pred, p_0_pred, k_0_pred, omega_0_pred, c_0_pred = model(torch.cat([x_0, y_0, t_0, Re_0, theta_0], dim=1))
    ic_losses = [
        torch.mean((u_0_pred - u_0) ** 2),
        torch.mean((v_0_pred - v_0) ** 2),
        torch.mean((p_0_pred - p_0) ** 2),
        torch.mean((k_0_pred - k_0) ** 2),
        torch.mean((omega_0_pred - omega_0) ** 2),
        torch.mean((c_0_pred - c_0) ** 2)
    ]
    loss_ic = sum([weights['ic'][i] * ic_losses[i] for i in range(len(ic_losses))])

    # Sparse data loss
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

    # Total loss
    total_loss = loss_pde + loss_bc + loss_ic + loss_sparse
    print([bc_losses[i].item() for i in range(len(bc_losses))])

    # Print loss components for debugging
    print(total_loss.item(), loss_bc.item(), loss_ic.item(), loss_sparse.item(), loss_pde.item())
    return total_loss, [pde_losses, bc_losses, ic_losses, sparse_losses]

def main():
    print('------------------------main------------------------------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # ---- input & output variable ranges for normalization & denormalization ----
    input_min = [-2.5, -2.5, 1, 30e6, 0]
    input_max = [7.5, 2.5, 100, 70e6, 2 * np.pi]

    output_min = [-1, -1, -2, 0, 0, 0]
    output_max = [2, 1, 1, 0.1, 30, 0.01]

    model = PINN(input_min, input_max, output_min, output_max).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Define a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    U_star = 9.0
    L_star = 80.0

    # Generate boundary and initial conditions
    boundary_conditions = generate_boundary_conditions(device)
    initial_conditions = generate_initial_conditions(device)

    # Generate sparse data
    sparse_data = generate_sparse_data(device)
    x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = sparse_data

    possible_Re_values = torch.tensor([9, 9, 9], dtype=torch.float32) * rho * L_star / mu

    # Default weights
    weights = default_weights()
    N3 = 1e5  # Update weights every N3 epochs

    # Set the number of collocation points for each loss part
    num_pde_points = 10
    num_bc_points = 1000
    num_ic_points = 10
    num_sparse_points = 10

    for epoch in range(500):
        # Randomly select a subset of collocation points for pde
        pde_indices = torch.randperm(len(x_sparse))[:num_pde_points]
        pde_data_subset = [d[pde_indices].unsqueeze(1) if len(d.shape) == 1 else d[pde_indices] for d in [x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse]]
        x_pde, y_pde, t_pde, Re_pde, theta_pde = pde_data_subset

        # Randomly select a subset of sparse data
        sparse_indices = torch.randperm(len(x_sparse))[:num_sparse_points]
        sparse_data_subset = [d[sparse_indices].unsqueeze(1) if len(d.shape) == 1 else d[sparse_indices] for d in sparse_data]

        # Subset boundary conditions to match the number of bc points
        bc_conditions_subset = []
        for bc in boundary_conditions:
            x_bc, y_bc, t_bc, Re_bc, theta_bc, conditions = bc
            bc_indices = torch.randperm(len(x_bc))[:num_bc_points]
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

        # Randomly select a subset of collocation points for ic
        ic_indices = torch.randperm(len(initial_conditions[0]))[:num_ic_points]
        initial_conditions_subset = [
            ic[ic_indices] for ic in initial_conditions
        ]

        loss_value = train_step(model, optimizer, x_pde, y_pde, t_pde, Re_pde, theta_pde,
                                bc_conditions_subset, initial_conditions_subset, sparse_data_subset, weights)

        if epoch % N3 == 0 and epoch != 0:
            new_weights = update_weights(model, x_pde, y_pde, t_pde, Re_pde, theta_pde,
                                         bc_conditions_subset, initial_conditions_subset, sparse_data_subset)
            for key in weights:
                weights[key] = 0.3 * weights[key] + 0.7 * new_weights[key]
            # Convert tensors to numpy arrays and format the values
            weights_str = {
                key: [f"{w.item():.0e}" for w in value.cpu().numpy()]
                for key, value in weights.items()
            }
            # Print the formatted values
            print("pde:", ", ".join(weights_str['pde']),
                  "bc:", ", ".join(weights_str['bc']),
                  "ic:", ", ".join(weights_str['ic']),
                  "sparse:", ", ".join(weights_str['sparse']))

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value}')
            save_model(model, 'pinn_model.pth')  # Save model at intervals

        scheduler.step()  # Update the learning rate

    # Plot the u field at time zero
    with torch.no_grad():
        # Use the first 13001 elements of sparse_data for prediction
        x_pred = x_sparse[:13001]
        y_pred = y_sparse[:13001]
        t_pred = t_sparse[:13001]
        Re_pred = Re_sparse[:13001]
        theta_pred = theta_sparse[:13001]

        check_tensor_stats(x_pred, 'x_pred')
        check_tensor_stats(y_pred, 'y_pred')
        check_tensor_stats(t_pred, 't_pred')
        check_tensor_stats(Re_pred, 'Re_pred')
        check_tensor_stats(theta_pred, 'theta_pred')

        # Predict values using the neural network
        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([x_pred, y_pred, t_pred, Re_pred, theta_pred], dim=1))

        check_tensor_stats(u_pred, 'u_pred')
        check_tensor_stats(v_pred, 'v_pred')
        check_tensor_stats(p_pred, 'p_pred')
        check_tensor_stats(k_pred, 'k_pred')
        check_tensor_stats(omega_pred, 'omega_pred')
        check_tensor_stats(c_pred, 'c_pred')

        # Convert predictions to numpy arrays for plotting
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        p_pred = p_pred.cpu().numpy()
        k_pred = k_pred.cpu().numpy()
        omega_pred = omega_pred.cpu().numpy()
        c_pred = c_pred.cpu().numpy()
        x_pred = x_pred.cpu()
        y_pred = y_pred.cpu()

        # Triangulation for plotting
        triang = tri.Triangulation(x_pred.squeeze(), y_pred.squeeze())

        # Mask the triangles inside the circle
        center = (0.0, 0.0)
        radius = 40.0 / L_star

        x_tri = x_pred[triang.triangles].mean(axis=1)
        y_tri = y_pred[triang.triangles].mean(axis=1)
        dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
        # Ensure the mask array has the same length as the number of triangles
        mask = dist_from_center < radius
        # Print debug information
        mask = mask.squeeze()  # Remove any extra dimensions
        mask = mask.cpu().numpy().astype(bool)
        print('-----------------MWMWMWMWMWMWMWMWMWMWMWMWM----------------')
        print(f"Length of mask: {len(mask)}")
        print(f"Length of triangles: {len(triang.triangles)}")
        print(f"Type of mask: {type(mask)}")
        print(f"Mask shape: {mask.shape}")
        print(f"Type of triang.triangles: {type(triang.triangles)}")
        print(f"triang.triangles shape: {triang.triangles.shape}")
        triang.set_mask(mask)

        # Plotting
        fig1 = plt.figure(figsize=(18, 12))

        plt.subplot(3, 2, 1)
        plt.tricontourf(triang, u_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $u$')
        plt.tight_layout()

        plt.subplot(3, 2, 2)
        plt.tricontourf(triang, v_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $v$')
        plt.tight_layout()

        plt.subplot(3, 2, 3)
        plt.tricontourf(triang, p_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $p$')
        plt.tight_layout()

        plt.subplot(3, 2, 4)
        plt.tricontourf(triang, k_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $k$')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, omega_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $omega$')
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.tricontourf(triang, c_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $c$')
        plt.tight_layout()

        # Save directory
        save_dir = os.path.join("figures")  # Adjust this path as necessary
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "code10.png"))

if __name__ == "__main__":
    main()

