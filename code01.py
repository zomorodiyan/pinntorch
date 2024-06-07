import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os


# Define a function to check if points are inside the circle
def inside_circle(x, y, center_x=0, center_y=0, radius=40):
    return (x - center_x)**2 + (y - center_y)**2 < radius**2

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.u_out = nn.Linear(20, 1)
        self.v_out = nn.Linear(20, 1)
        self.p_out = nn.Linear(20, 1)
        self.k_out = nn.Linear(20, 1)
        self.omega_out = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        u = self.u_out(x)
        v = self.v_out(x)
        p = self.p_out(x)
        k = self.k_out(x)
        omega = self.omega_out(x)
        return u, v, p, k, omega

# Define the PDE residuals for the SST k-omega model
def pde_residuals(model, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u, v, p, k, omega = model(torch.cat([x, y, t], dim=1))

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

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    k_xx = torch.autograd.grad(k_x, x, grad_outputs=torch.ones_like(k_x), create_graph=True)[0]
    k_yy = torch.autograd.grad(k_y, y, grad_outputs=torch.ones_like(k_y), create_graph=True)[0]
    omega_xx = torch.autograd.grad(omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_yy = torch.autograd.grad(omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]

    continuity_residual = u_x + v_y

    momentum_x_residual = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y_residual = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    k_residual = k_t + u * k_x + v * k_y - beta_star * k * omega + sigma_k * (k_xx + k_yy) - source_k(x, y, t)
    omega_residual = omega_t + u * omega_x + v * omega_y - beta * omega ** 2 + sigma_omega * (omega_xx + omega_yy) - gamma * k * omega + source_omega(x, y, t)

    return continuity_residual, momentum_x_residual, momentum_y_residual, k_residual, omega_residual

# Define the loss function
def loss(model, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega, x_b, y_b, t_b, x_0, y_0, u_0, v_0, p_0, k_0, omega_0):
    continuity_residual, momentum_x_residual, momentum_y_residual, k_residual, omega_residual = pde_residuals(model, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega)

    loss_continuity = torch.mean(continuity_residual**2)
    loss_momentum_x = torch.mean(momentum_x_residual**2)
    loss_momentum_y = torch.mean(momentum_y_residual**2)
    loss_k = torch.mean(k_residual**2)
    loss_omega = torch.mean(omega_residual**2)

    u_b, v_b, p_b, k_b, omega_b = model(torch.cat([x_b, y_b, t_b], dim=1))
    loss_bc = torch.mean(u_b**2) + torch.mean(v_b**2) + torch.mean(p_b**2) + torch.mean(k_b**2) + torch.mean(omega_b**2)  # Adjust for actual BC conditions

    u_0_pred, v_0_pred, p_0_pred, k_0_pred, omega_0_pred = model(torch.cat([x_0, y_0, torch.zeros_like(x_0)], dim=1))
    loss_ic = torch.mean((u_0_pred - u_0)**2) + torch.mean((v_0_pred - v_0)**2) + torch.mean((p_0_pred - p_0)**2) + torch.mean((k_0_pred - k_0)**2) + torch.mean((omega_0_pred - omega_0)**2)

    total_loss = loss_continuity + loss_momentum_x + loss_momentum_y + loss_k + loss_omega + loss_bc + loss_ic
    return total_loss

# Define the training step
def train_step(model, optimizer, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega, x_b, y_b, t_b, x_0, y_0, u_0, v_0, p_0, k_0, omega_0):
    optimizer.zero_grad()
    total_loss = loss(model, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega, x_b, y_b, t_b, x_0, y_0, u_0, v_0, p_0, k_0, omega_0)
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# Define the main function to train the PINN
def main():
    # Check if GPU is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # model and physical constants
    nu = 0.01
    beta_star = 0.09
    beta = 0.075
    sigma_k = 0.5
    sigma_omega = 0.5
    gamma = 0.1

    # Generate collocation points (x, y, t)
    x = torch.linspace(-200, 600, 1000)
    y = torch.linspace(-200, 200, 500)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    x_flat = x_grid.flatten().unsqueeze(1).to(device)
    y_flat = y_grid.flatten().unsqueeze(1).to(device)
    t_flat = torch.rand_like(x_flat).to(device)

    # Filter out points inside the circle
    mask = ~inside_circle(x_flat, y_flat)
    x = x_flat[mask].to(device)
    y = y_flat[mask].to(device)
    t = t_flat[mask].to(device)
    print('x.shape')
    print(x.shape)

    # Ensure x, y, and t are 2D tensors with shape [N, 1]
    x = x.view(-1, 1).to(device)
    y = y.view(-1, 1).to(device)
    t = t.view(-1, 1).to(device)

    # Generate boundary points (x_b, y_b, t_b)
    theta = torch.linspace(0, 2 * np.pi, 200).to(device)
    x_b = (40 * torch.cos(theta)).view(-1, 1).to(device)
    y_b = (40 * torch.sin(theta)).view(-1, 1).to(device)
    t_b = torch.zeros_like(x_b).to(device)

    # Generate initial condition points (x_0, y_0, u_0, v_0, p_0, k_0, omega_0)
    x_0 = torch.linspace(-200, 600, 200).to(device)
    y_0 = torch.linspace(-200, 200, 100).to(device)
    x_0, y_0 = torch.meshgrid(x_0, y_0, indexing='ij')
    x_0 = x_0.flatten().unsqueeze(1).to(device)
    y_0 = y_0.flatten().unsqueeze(1).to(device)
    u_0 = torch.zeros_like(x_0).to(device)
    v_0 = torch.zeros_like(x_0).to(device)
    p_0 = torch.zeros_like(x_0).to(device)
    k_0 = torch.zeros_like(x_0).to(device)
    omega_0 = torch.zeros_like(x_0).to(device)

    # Define source terms (placeholder functions)
    def source_k(x, y, t):
        return torch.zeros_like(x)

    def source_omega(x, y, t):
        return torch.zeros_like(x)

    for epoch in range(1000):
        loss_value = train_step(model, optimizer, x, y, t, nu, beta_star, beta, sigma_k, sigma_omega, gamma, source_k, source_omega, x_b, y_b, t_b, x_0, y_0, u_0, v_0, p_0, k_0, omega_0)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value}')
        # Plot the u field at time zero
    #------------------------visualization------------------------
    with torch.no_grad():
        # Generate a grid of points for prediction
        x_pred = torch.linspace(-200, 600, 200).to(device)
        y_pred = torch.linspace(-200, 200, 200).to(device)
        t_pred = torch.zeros(40000).to(device)  # Predicting at time t = 0

        xx, yy = torch.meshgrid(x_pred, y_pred, indexing='ij')
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        tt = torch.zeros_like(xx_flat)  # Predicting at time t = 0

        # Filter out points inside the circle
        mask = ~inside_circle(xx_flat, yy_flat)
        xx_masked = xx_flat[mask]
        yy_masked = yy_flat[mask]
        tt_masked = tt[mask]

        # Predict values using the neural network
        u_pred, v_pred, p_pred, k_pred, omega_pred = model(torch.cat([xx_masked.unsqueeze(1), yy_masked.unsqueeze(1), tt_masked.unsqueeze(1)], dim=1))

# Convert predictions to numpy arrays for plotting
        u_pred = u_pred.cpu().numpy()
        v_pred = v_pred.cpu().numpy()
        p_pred = p_pred.cpu().numpy()
        k_pred = k_pred.cpu().numpy()
        omega_pred = omega_pred.cpu().numpy()
        xx_masked = xx_masked.cpu()
        yy_masked = yy_masked.cpu()

# Triangulation for plotting
        triang = tri.Triangulation(xx_masked, yy_masked)

# Mask the triangles inside the circle
        center = (0.0, 0.0)
        radius = 40.0

        x_tri = xx_masked[triang.triangles].mean(axis=1)
        y_tri = yy_masked[triang.triangles].mean(axis=1)
        dist_from_center = np.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
        triang.set_mask(dist_from_center < radius)

# print sizes
        print('x_tri.shape,y_tri.shape,u_pred.shape')
        print(x_tri.shape,y_tri.shape,u_pred.shape)

# Plotting
        fig1 = plt.figure(figsize=(18, 12))

        plt.subplot(3, 1, 1)
        plt.tricontourf(triang, u_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $u$')
        plt.tight_layout()

        plt.subplot(3, 1, 2)
        plt.tricontourf(triang, v_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $v$')
        plt.tight_layout()

        plt.subplot(3, 1, 3)
        plt.tricontourf(triang, p_pred.squeeze(), cmap='jet', levels=100)
        plt.colorbar()
        plt.title('Predicted $p$')
        plt.tight_layout()

# Save directory
        save_dir = os.path.join("figures")  # Adjust this path as necessary
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "code01.png"))
if __name__ == "__main__":
    main()
