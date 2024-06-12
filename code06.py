import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch.nn.functional as F
from utils import calculate_range_and_divergence

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
    print('coords_shape: ', coords.shape)

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

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(5, 20)  # 5 inputs: x, y, t, Re, theta
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.u_out = nn.Linear(20, 1)
        self.v_out = nn.Linear(20, 1)
        self.p_out = nn.Linear(20, 1)
        self.k_out = nn.Linear(20, 1)
        self.omega_out = nn.Linear(20, 1)
        self.c_out = nn.Linear(20, 1)  # Output for concentration

        self._initialize_weights()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        u = self.u_out(x)
        v = self.v_out(x)
        p = self.p_out(x)
        k = self.k_out(x)
        omega = self.omega_out(x)
        c = self.c_out(x)
        return u, v, p, k, omega, c

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

def safe_sqrt(tensor, epsilon=1e-10):
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
    Re.requires_grad_(True)
    u, v, p, k, omega, c = model(torch.cat([x, y, t, Re, theta], dim=1))


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
    #k = ensure_positive(k) # smoothly clips negative values
    phi_11 = safe_sqrt(k) / (0.09 * omega * y_hat)
    phi_12 = 500 / (Re * y_hat ** 2 * omega)
    phi_13 = 4 * k / (sigma_omega2 * D_omega_plus * y_hat ** 2)
    phi_1 = smooth_minimum(smooth_maximum(phi_11, phi_12), phi_13)
    phi_21 = (2 * safe_sqrt(k)) / (0.09 * omega * y_hat)
    phi_22 = 500 / (Re * y_hat ** 2 * omega)
    phi_2 = smooth_maximum(phi_21, phi_22)

    F1 = torch.tanh(phi_1 ** 4)
    F2 = torch.tanh(phi_2)
    beta_i = F1 * beta_i1 + (1 - F1) * beta_i2
    alpha_star_0 = beta_i / 3
    alpha_infinity_1 = beta_i1 / beta_star_infinity - kappa **\
            2 / (sigma_omega1 * (beta_star_infinity)**0.5)
    alpha_infinity_2 = beta_i2 / beta_star_infinity - kappa **\
            2 / (sigma_omega2 * (beta_star_infinity)**0.5)
    alpha_infinity = F1 * alpha_infinity_1 + (1 - F1) * alpha_infinity_2

    Re_t = k / (mu * omega)
    alpha_star = alpha_star_infinity * (alpha_star_0 + Re_t / R_k) / (1 + Re_t / R_k)
    alpha = (alpha_infinity / alpha_star) * ((alpha_0 + Re_t / R_omega) / (1 + Re_t / R_omega))
    beta_star_i = beta_star_infinity * ((4 / 15 + (Re_t / R_beta) ** 4) / (1 + (Re_t / R_beta) ** 4))
    M_t = U_star * safe_sqrt(2 * k / (gamma * R * T))
    F_Mt = smooth_conditional(M_t<=M_t0, torch.zeros_like(M_t), M_t ** 2 - M_t0 ** 2)

    beta_star = beta_star_i * (1 + xi_star * F_Mt)
    beta = beta_i * (1 - beta_star_i / beta_i * xi_star * F_Mt)
    sigma_k = 1 / (F1 / sigma_k1 + (1 - F1) / sigma_k2)
    sigma_omega = 1 / (F1 / sigma_omega1 + (1 - F1) / sigma_omega2)
    S = safe_sqrt(2 * ((u_x) ** 2 + (v_y) ** 2 + 0.5 * (u_y + v_x) ** 2))

    mu_t = k / omega * (1 / smooth_maximum(1 / alpha_star, S * F2 / (a1 * omega)))
    G_k = mu_t * S ** 2
    Y_k = beta_star * k * omega
    G_k_tilde = smooth_minimum(G_k, 10 * beta_star * k * omega)

    G_omega = alpha / mu_t * G_k_tilde
    Y_omega = beta * omega ** 2
    D_omega = 2 * (1 - F1) * (sigma_omega2 / omega) * (k_x * omega_x + k_y * omega_y)

    continuity_residual = u_x + v_y
    x_mom_gradx = torch.autograd.grad((1/Re+mu_t)*(4/3*u_x-2/3*v_y), x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_mom_grady = torch.autograd.grad((1/Re+mu_t)*(v_x+u_y), y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    x_momentum_residual = u_t + u * u_x + v * u_y + p_x - x_mom_gradx - x_mom_grady

    y_mom_grady = torch.autograd.grad((1/Re+mu_t)*(4/3*v_y-2/3*u_x), y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_mom_gradx = torch.autograd.grad((1/Re+mu_t)*(v_x+u_y), x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    y_momentum_residual = v_t + u * v_x + v * v_y + p_y - y_mom_grady - y_mom_gradx

    k_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_x, x, grad_outputs=torch.ones_like(k_x), create_graph=True)[0]
    k_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_k) * k_y, y, grad_outputs=torch.ones_like(k_y), create_graph=True)[0]
    k_residual = k_t + u * k_x + v * k_y - k_transport_term1 - k_transport_term2 - G_k + Y_k
    omega_transport_term1 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_transport_term2 = torch.autograd.grad((1 / Re + mu_t / sigma_omega) * omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]
    omega_residual = omega_t + u * omega_x + v * omega_y - omega_transport_term1 - omega_transport_term2 - G_omega + Y_omega - D_omega
    c_residual = c_t + u * c_x + v * c_y - (1 / Re) * (c_x + c_y)  # Convection-diffusion equation

    return continuity_residual, x_momentum_residual, y_momentum_residual, k_residual, omega_residual, c_residual


# Define the loss function
def loss(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data):
    # PDE residuals
    continuity_residual, x_momentum_residual, y_momentum_residual, k_residual, omega_residual, c_residual = pde_residuals(model, x, y, t, Re, theta)

    loss_continuity = torch.mean(continuity_residual ** 2)
    loss_x_momentum = torch.mean(x_momentum_residual ** 2)
    loss_y_momentum = torch.mean(y_momentum_residual ** 2)
    loss_k = torch.mean(k_residual ** 2)
    loss_omega = torch.mean(omega_residual ** 2)
    loss_c = torch.mean(c_residual ** 2)

    total_loss = loss_continuity + loss_x_momentum + loss_y_momentum + loss_k + loss_omega + loss_c
    total_loss_pde = loss_continuity + loss_x_momentum + loss_y_momentum + loss_k + loss_omega + loss_c

    total_loss_bc = 0
    # Boundary conditions
    for bc in boundary_conditions:
        x_b, y_b, t_b, Re_b, theta_b, conditions = bc
        x_b.requires_grad_(True)
        y_b.requires_grad_(True)
        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([x_b, y_b, t_b, Re_b, theta_b], dim=1))

        loss_one_bc = 0 # initialize total_loss_bc
        for variable, condition in conditions.items():
            if condition['type'] == 'Dirichlet':
                value = condition['value']
                if variable == 'u':
                    loss_bc = torch.mean((u_pred - value) ** 2)
                elif variable == 'v':
                    loss_bc = torch.mean((v_pred - value) ** 2)
                elif variable == 'p':
                    loss_bc = torch.mean((p_pred - value) ** 2)
                elif variable == 'k':
                    loss_bc = torch.mean((k_pred - value) ** 2)
                elif variable == 'omega':
                    loss_bc = torch.mean((omega_pred - value) ** 2)
                elif variable == 'c':
                    loss_bc = torch.mean((c_pred - value) ** 2)
            elif condition['type'] == 'Neumann':
                dir_deriv = condition['dir_deriv']
                value = condition['value']
                if dir_deriv == 'x':
                    if variable == 'u':
                        deriv = torch.autograd.grad(u_pred, x_b, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
                    elif variable == 'v':
                        deriv = torch.autograd.grad(v_pred, x_b, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
                    elif variable == 'p':
                        deriv = torch.autograd.grad(p_pred, x_b, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
                    elif variable == 'k':
                        deriv = torch.autograd.grad(k_pred, x_b, grad_outputs=torch.ones_like(k_pred), create_graph=True)[0]
                    elif variable == 'omega':
                        deriv = torch.autograd.grad(omega_pred, x_b, grad_outputs=torch.ones_like(omega_pred), create_graph=True)[0]
                    elif variable == 'c':
                        deriv = torch.autograd.grad(c_pred, x_b, grad_outputs=torch.ones_like(c_pred), create_graph=True)[0]
                elif dir_deriv == 'y':
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
                    elif variable == 'c':
                        deriv = torch.autograd.grad(c_pred, y_b, grad_outputs=torch.ones_like(c_pred), create_graph=True)[0]
                loss_bc = torch.mean((deriv - value) ** 2)
            loss_one_bc += loss_bc
        total_loss_bc += loss_one_bc
    total_loss += total_loss_bc

    # Initial conditions
    x_0, y_0, t_0, Re_0, theta_0, u_0, v_0, p_0, k_0, omega_0, c_0 = initial_conditions
    u_0_pred, v_0_pred, p_0_pred, k_0_pred, omega_0_pred, c_0_pred = model(torch.cat([x_0, y_0, t_0, Re_0, theta_0], dim=1))
    loss_ic = torch.mean((u_0_pred - u_0) ** 2) + torch.mean((v_0_pred - v_0) ** 2) + torch.mean((p_0_pred - p_0) ** 2) + torch.mean((k_0_pred - k_0) ** 2) + torch.mean((omega_0_pred - omega_0) ** 2) + torch.mean((c_0_pred - c_0) ** 2)
    total_loss += loss_ic



    # Sparse data
    x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse = sparse_data
    u_sparse_pred, v_sparse_pred, p_sparse_pred, k_sparse_pred, omega_sparse_pred, c_sparse_pred = model(torch.cat([x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse], dim=1))
    loss_sparse = torch.mean((u_sparse_pred - u_sparse) ** 2) + torch.mean((v_sparse_pred - v_sparse) ** 2) + torch.mean((p_sparse_pred - p_sparse) ** 2) + torch.mean((k_sparse_pred - k_sparse) ** 2) + torch.mean((omega_sparse_pred - omega_sparse) ** 2) + torch.mean((c_sparse_pred - c_sparse) ** 2)
    total_loss += loss_sparse

    print('....................  loss values  .......................')
    print('total_loss, loss_sparse, loss_ic, loss_bc, total_loss_pde')
    print(total_loss.item(), loss_sparse.item(), loss_ic.item(),
          loss_bc.item(), total_loss_pde.item())

    return total_loss

# Define the training step
def train_step(model, optimizer, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data):
    optimizer.zero_grad()
    total_loss = loss(model, x, y, t, Re, theta, boundary_conditions, initial_conditions, sparse_data)
    total_loss.backward()
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
    k_ref = np.clip(k_ref, 0.0, None)
    omega_ref = np.clip(omega_ref, 0.0, 2*L_star/U_star)
    c_ref = np.clip(c_ref, 0.0, 0.001)

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
    omega_in_value = (3/2*(0.05*9.0)**2/9.0**2)/(10*1.7894e-5/1.225) # for now

    x_in = torch.full((400, 1), -200.0).to(device)
    y_in = torch.linspace(-200, 200, 400).view(-1, 1).to(device)
    t_in = torch.zeros_like(x_in).to(device)
    Re_in = torch.full((400, 1), Re_medium).to(device) # for now
    theta_in = torch.zeros_like(x_in).to(device)
    u_in = torch.full((400, 1), 1.0).to(device)
    v_in = torch.zeros_like(u_in).to(device)
    k_in = torch.full((400, 1), k_in_value).to(device) # for now
    omega_in = torch.full((400, 1), omega_in_value).to(device) # for now
    c_in = torch.zeros_like(x_in).to(device)  # Concentration

    # Symmetry (Top and Bottom Boundaries: -200 ≤ x ≤ 600, y = ±200)
    x_sym = torch.linspace(-200, 600, 400).view(-1, 1).to(device)
    y_sym_top = torch.full((400, 1), 200.0).to(device)
    y_sym_bottom = torch.full((400, 1), -200.0).to(device)
    t_sym = torch.zeros_like(x_sym).to(device)
    Re_sym = torch.full((400, 1), Re_medium).to(device)
    theta_sym = torch.zeros_like(x_sym).to(device)
    u_sym = torch.zeros_like(x_sym).to(device)
    v_sym = torch.zeros_like(x_sym).to(device) # for both neumann and dirichlet boundary conditions
    p_sym = torch.zeros_like(x_sym).to(device)
    k_sym = torch.zeros_like(x_sym).to(device)
    omega_sym = torch.zeros_like(x_sym).to(device)

    # Constant Pressure Outlet (Right Boundary: x = 600, -200 ≤ y ≤ 200)
    x_out = torch.full((400, 1), 600.0).to(device)
    y_out = torch.linspace(-200, 200, 400).view(-1, 1).to(device)
    t_out = torch.zeros_like(x_out).to(device)
    Re_out = torch.full((400, 1), Re_medium).to(device)
    theta_out = torch.zeros_like(x_out).to(device)
    u_out = torch.zeros_like(x_out).to(device)
    v_out = torch.zeros_like(x_out).to(device)
    p_out = torch.zeros((400, 1)).to(device)  # Dirichlet p = 0
    k_out = torch.zeros_like(x_out).to(device)
    omega_out = torch.zeros_like(x_out).to(device)
    c_out = torch.zeros_like(x_out).to(device)

    # Circular Wall (Radius 40, Centered at (0,0))
    theta_rand = torch.linspace(0, 2 * np.pi, 200).to(device)
    x_wall = (40 * torch.cos(theta_rand)).view(-1, 1).to(device)
    y_wall = (40 * torch.sin(theta_rand)).view(-1, 1).to(device)
    t_wall = torch.zeros_like(x_wall).to(device)
    Re_wall = torch.full((200, 1), Re_medium).to(device)
    theta_wall = torch.zeros_like(x_wall).to(device)
    u_wall = torch.zeros_like(x_wall).to(device)
    v_wall = torch.zeros_like(x_wall).to(device)
    k_wall = torch.zeros_like(x_wall).to(device)  # u, v, k are zero
    omega_wall = torch.ones_like(x_wall).to(device)  # Specified value
    c_wall = torch.zeros_like(x_wall).to(device)  # Concentration

    boundary_conditions = [
        (x_in, y_in, t_in, Re_in, theta_in, {
            'u': {'type': 'Dirichlet', 'value': u_in},
            'v': {'type': 'Dirichlet', 'value': v_in},
            'k': {'type': 'Dirichlet', 'value': k_in},
            'omega': {'type': 'Dirichlet', 'value': omega_in},
            'c': {'type': 'Dirichlet', 'value': c_in}
        }),
        (x_sym, y_sym_top, t_sym, Re_sym, theta_sym, {
            'u': {'type': 'Neumann', 'dir_deriv': 'y', 'value': u_sym},
            'v': {'type': 'Dirichlet', 'value': v_sym},
            'v': {'type': 'Neumann', 'dir_deriv': 'y', 'value': v_sym},
            'p': {'type': 'Neumann', 'dir_deriv': 'y', 'value': p_sym},
            'k': {'type': 'Neumann', 'dir_deriv': 'y', 'value': k_sym},
            'omega': {'type': 'Neumann', 'dir_deriv': 'y', 'value': omega_sym}
        }),
        (x_sym, y_sym_bottom, t_sym, Re_sym, theta_sym, {
            'u': {'type': 'Neumann', 'dir_deriv': 'y', 'value': u_sym},
            'v': {'type': 'Dirichlet', 'value': v_sym},
            'v': {'type': 'Neumann', 'dir_deriv': 'y', 'value': v_sym},
            'p': {'type': 'Neumann', 'dir_deriv': 'y', 'value': p_sym},
            'k': {'type': 'Neumann', 'dir_deriv': 'y', 'value': k_sym},
            'omega': {'type': 'Neumann', 'dir_deriv': 'y', 'value': omega_sym}
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
    x_sparse = torch.tensor(coords[:, 0], dtype=torch.float32).repeat(u_ref.shape[0]).unsqueeze(1).to(device)
    y_sparse = torch.tensor(coords[:, 1], dtype=torch.float32).repeat(u_ref.shape[0]).unsqueeze(1).to(device)
    t_sparse = torch.arange(1.0, u_ref.shape[0] + 1.0).\
        repeat_interleave(u_ref.shape[1]).unsqueeze(1).to(device)* (U_star/L_star)
    Re_sparse = torch.full_like(x_sparse, Re.item()).to(device)
    theta_sparse = torch.zeros_like(x_sparse).to(device)
    u_sparse = torch.tensor(u_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    v_sparse = torch.tensor(v_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    p_sparse = torch.tensor(p_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    k_sparse = torch.tensor(k_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    omega_sparse = torch.tensor(omega_ref, dtype=torch.float32).reshape(-1, 1).to(device)
    c_sparse = torch.tensor(c_ref, dtype=torch.float32).reshape(-1, 1).to(device)

    print('---------------------u_sparse--shape--------------------------')
    print(u_sparse.shape)
    print(v_sparse.shape)
    print(k_sparse.shape)
    print(p_sparse.shape)
    print(omega_sparse.shape)
    print(c_sparse.shape)
    return x_sparse, y_sparse, t_sparse, Re_sparse, theta_sparse, u_sparse, v_sparse, p_sparse, k_sparse, omega_sparse, c_sparse

# Define the main function to train the PINN
def main():
    print('------------------------main------------------------------')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    '''
    u_ref, v_ref, p_ref, k_ref, omega_ref, coords,\ inflow_coords, outflow_coords, symmetry_coords, cylinder_coords, nu\
      = get_dataset()
    '''

    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    U_star = 9.0
    L_star = 80.0
    Re = rho * U_star * L_star / mu
    # Generate collocation points (x, y, t, Re, theta)
    x = torch.linspace(-200/L_star, 600/L_star, 1000).to(device)
    y = torch.linspace(-200/L_star, 200/L_star, 500).to(device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    x_flat = x_grid.flatten().unsqueeze(1).to(device)
    y_flat = y_grid.flatten().unsqueeze(1).to(device)
    t_flat = torch.rand_like(x_flat).to(device)*100*U_star/L_star
    Re_flat = torch.full_like(x_flat, Re).to(device)
    theta_flat = torch.zeros_like(x_flat).to(device)

    # Filter out points inside the circle
    mask = ~inside_circle(x_flat, y_flat)
    x = x_flat[mask].to(device)
    y = y_flat[mask].to(device)
    t = t_flat[mask].to(device)
    Re = Re_flat[mask].to(device)
    theta = theta_flat[mask].to(device)

    # Ensure x, y, t, Re, and theta are 2D tensors with shape [N, 1]
    x = x.view(-1, 1).to(device)
    y = y.view(-1, 1).to(device)
    t = t.view(-1, 1).to(device)
    Re = Re.view(-1, 1).to(device)
    theta = theta.view(-1, 1).to(device)

    # Generate boundary and initial conditions
    boundary_conditions= generate_boundary_conditions(device)
    initial_conditions = generate_initial_conditions(device)

    # Generate sparse data
    sparse_data = generate_sparse_data(device)


    for epoch in range(1000):
        # Randomly select a subset of collocation points and sparse data for this training step
        subset_indices = torch.randperm(len(x))[:1000]
        x_subset = x[subset_indices]
        y_subset = y[subset_indices]
        t_subset = t[subset_indices]
        Re_subset = Re[subset_indices]
        theta_subset = theta[subset_indices]
        sparse_data_subset = [d.squeeze()[subset_indices].unsqueeze(1) for d in sparse_data]
        loss_value = train_step(model, optimizer, x_subset, y_subset, t_subset, Re_subset, theta_subset, boundary_conditions, initial_conditions, sparse_data_subset)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value}')
            save_model(model, 'pinn_model.pth')  # Save model at intervals

    # Plot the u field at time zero
    with torch.no_grad():
        # Generate a grid of points for prediction
        x_pred = torch.linspace(-200/80, 600/80, 200).to(device)
        y_pred = torch.linspace(-200/80, 200/80, 200).to(device)
        t_pred = torch.zeros(40000).to(device)  # Predicting at time t = 0
        Re_pred = torch.full_like(t_pred, Re_medium).to(device)
        theta_pred = torch.zeros_like(t_pred).to(device)

        xx, yy = torch.meshgrid(x_pred, y_pred, indexing='ij')
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        tt = torch.zeros_like(xx_flat)  # Predicting at time t = 0
        Re_flat = torch.full_like(tt, Re_medium).to(device)
        theta_flat = torch.zeros_like(tt).to(device)

        # Filter out points inside the circle
        mask = ~inside_circle(xx_flat*L_star, yy_flat*L_star)
        xx_masked = xx_flat[mask]
        yy_masked = yy_flat[mask]
        tt_masked = tt[mask]
        Re_masked = Re_flat[mask]
        theta_masked = theta_flat[mask]

        # Predict values using the neural network
        u_pred, v_pred, p_pred, k_pred, omega_pred, c_pred = model(torch.cat([xx_masked.unsqueeze(1), yy_masked.unsqueeze(1), tt_masked.unsqueeze(1), Re_masked.unsqueeze(1), theta_masked.unsqueeze(1)], dim=1))

        # Convert predictions to numpy arrays for plotting
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        p_pred = p_pred.cpu().numpy()
        k_pred = k_pred.cpu().numpy()
        omega_pred = omega_pred.cpu().numpy()
        c_pred = c_pred.cpu().numpy()
        xx_masked = xx_masked.cpu()
        yy_masked = yy_masked.cpu()

        # Triangulation for plotting
        triang = tri.Triangulation(xx_masked, yy_masked)

        # Mask the triangles inside the circle
        center = (0.0, 0.0)
        radius = 40.0/L_star

        x_tri = xx_masked[triang.triangles].mean(axis=1)
        y_tri = yy_masked[triang.triangles].mean(axis=1)
        dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
        triang.set_mask(dist_from_center < radius)

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
        plt.title('Predicted $p$')
        plt.tight_layout()

        plt.subplot(3, 2, 5)
        plt.tricontourf(triang, omega_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $p$')
        plt.tight_layout()

        # Save directory
        save_dir = os.path.join("figures")  # Adjust this path as necessary
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "code06.png"))

if __name__ == "__main__":
    main()
