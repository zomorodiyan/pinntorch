# PINN-based Turbulent Flow Simulation

This repository contains code for a Physics-Informed Neural Network (PINN) designed to model turbulent flows using the SST k-omega model and convection-diffusion equation. The code is written in Python and utilizes PyTorch for deep learning operations.

## Code Overview

The main function that sets up the training environment, initializes the model, and performs training iterations:
- Uses CUDA for GPU acceleration if available.
- Defines the loss criterion, optimizer, and learning rate scheduler.
- Implements a training loop with logging and weight updates.

## Running the Code

1. Ensure you have Python and PyTorch installed.
2. Load the dataset (`data/single.npy`).
   or generate it from dataset (`data/unsteady_emh.npy`):
      ```sh
   python preprocess.py
      
4. From src directory run the script using the command:
   ```sh
   python code38.py
   
## Dependencies

- Python 3.6+
- PyTorch
- Numpy

## Acknowledgments

This work is based on the Physics-Informed Neural Networks (PINNs) approach for solving PDEs and simulating turbulent flows using the SST k-omega model.

## Data Files

The data files required for running the code are available on OneDrive. You can download them using the following link (it requires access):

[south west leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_swmh.npy?csf=1&web=1&e=t6ohmI)
[west leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_wmh.npy?csf=1&web=1&e=3uri4a)
[south leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_smh.npy?csf=1&web=1&e=g5pbew)
[north leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_nmh.npy?csf=1&web=1&e=sSkcRs)
[north_west leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_nwmh.npy?csf=1&web=1&e=fnOvNS)
[east leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_emh.npy?csf=1&web=1&e=tqePXt)
[north_east leak](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/unsteady_nemh.npy?csf=1&web=1&e=VGGfsW)
[single](https://ostatemailokstate-my.sharepoint.com/:u:/r/personal/mehrdad_zomorodiyan_okstate_edu/Documents/pinn_data/single.npy?csf=1&web=1&e=J58mUe)

After downloading, place the files in the `data` directory of the repository.
