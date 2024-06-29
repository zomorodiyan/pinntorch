# PINN-based Turbulent Flow Simulation

This repository contains code for a Physics-Informed Neural Network (PINN) designed to model turbulent flows using the SST k-omega model and convection-diffusion equation. The code is written in Python and utilizes PyTorch for deep learning operations.

## Code Overview

### `generate_boundary_conditions`

This function generates boundary conditions for different regions in the simulation domain:
- `interval`: The current interval in the simulation.
- `num_samples`: The number of samples for boundary conditions.

It returns a list of boundary conditions for the inlet, symmetry, outlet, and wall.

### `PINN` Class

Defines the architecture of the Physics-Informed Neural Network:
- Layers: A normalization layer, a Fourier embedding layer, and several fully connected layers.
- `forward`: Defines the forward pass of the network.
- `_initialize_weights` and `initialize_weights`: Methods for initializing weights.

### `NormalizationLayer` and `DenormalizationLayer`

These classes handle normalization and denormalization of input and output data to ensure the network operates within a suitable range.

### `FourierEmbedding` Class

Applies a Fourier embedding to the input features to capture periodic patterns in the data.

### `smooth_maximum`, `smooth_minimum`, and `smooth_conditional`

Utility functions for smooth approximations of max, min, and conditional operations.

### `safe_sqrt`

A safe square root function that adds a small epsilon to avoid taking the square root of zero.

### `pde_residuals`

Defines the Partial Differential Equation (PDE) residuals for the SST k-omega model and the convection-diffusion equation. It calculates the residuals for continuity, momentum, turbulence kinetic energy (k), and specific dissipation rate (omega).

### `bc_calc_loss`

Calculates the loss for the boundary conditions by comparing the predicted and actual values.

### `CustomDataset` Class

Defines a custom dataset for loading and managing simulation data:
- `data_array`: The input data array.
- `num_intervals`: Number of intervals in the simulation.
- `num_simulations`: Number of simulations in the dataset.

### `prepare_inputs_outputs`

Prepares the inputs and outputs for the model from batch data.

### `calculate_ic_losses`

Calculates the initial condition losses by comparing model predictions with actual outputs.

### `main`

The main function that sets up the training environment, initializes the model, and performs training iterations:
- Uses CUDA for GPU acceleration if available.
- Defines the loss criterion, optimizer, and learning rate scheduler.
- Implements a training loop with logging and weight updates.

## Running the Code

1. Ensure you have Python and PyTorch installed.
2. Load the dataset (`data/preprocessed_clipped.npy`).
3. Run the script using the command:
   ```sh
   python main.py

txt
Copy code
# PINN-based Turbulent Flow Simulation

This repository contains code for a Physics-Informed Neural Network (PINN) designed to model turbulent flows using the SST k-omega model and convection-diffusion equation. The code is written in Python and utilizes PyTorch for deep learning operations.

## Code Overview

### `generate_boundary_conditions`

This function generates boundary conditions for different regions in the simulation domain:
- `interval`: The current interval in the simulation.
- `num_samples`: The number of samples for boundary conditions.

It returns a list of boundary conditions for the inlet, symmetry, outlet, and wall.

### `PINN` Class

Defines the architecture of the Physics-Informed Neural Network:
- Layers: A normalization layer, a Fourier embedding layer, and several fully connected layers.
- `forward`: Defines the forward pass of the network.
- `_initialize_weights` and `initialize_weights`: Methods for initializing weights.

### `NormalizationLayer` and `DenormalizationLayer`

These classes handle normalization and denormalization of input and output data to ensure the network operates within a suitable range.

### `FourierEmbedding` Class

Applies a Fourier embedding to the input features to capture periodic patterns in the data.

### `smooth_maximum`, `smooth_minimum`, and `smooth_conditional`

Utility functions for smooth approximations of max, min, and conditional operations.

### `safe_sqrt`

A safe square root function that adds a small epsilon to avoid taking the square root of zero.

### `pde_residuals`

Defines the Partial Differential Equation (PDE) residuals for the SST k-omega model and the convection-diffusion equation. It calculates the residuals for continuity, momentum, turbulence kinetic energy (k), and specific dissipation rate (omega).

### `bc_calc_loss`

Calculates the loss for the boundary conditions by comparing the predicted and actual values.

### `CustomDataset` Class

Defines a custom dataset for loading and managing simulation data:
- `data_array`: The input data array.
- `num_intervals`: Number of intervals in the simulation.
- `num_simulations`: Number of simulations in the dataset.

### `prepare_inputs_outputs`

Prepares the inputs and outputs for the model from batch data.

### `calculate_ic_losses`

Calculates the initial condition losses by comparing model predictions with actual outputs.

### `main`

The main function that sets up the training environment, initializes the model, and performs training iterations:
- Uses CUDA for GPU acceleration if available.
- Defines the loss criterion, optimizer, and learning rate scheduler.
- Implements a training loop with logging and weight updates.

## Running the Code

1. Ensure you have Python and PyTorch installed.
2. Load the dataset (`data/preprocessed_clipped.npy`).
3. Run the script using the command:
   ```sh
   python main.py
## Dependencies
- Python 3.6+
- PyTorch
- Numpy
## Acknowledgments
This work is based on the Physics-Informed Neural Networks (PINNs) approach for solving PDEs and simulating turbulent flows using the SST k-omega model.

For any issues or questions, please open an issue or contact the repository owner.
