# PINN-based Turbulent Flow Simulation

This repository contains code for a Physics-Informed Neural Network (PINN) designed to model turbulent flows using the SST k-omega model and convection-diffusion equation. The code is written in Python and utilizes PyTorch for deep learning operations.

## Code Overview

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

## Data Files

The data files required for running the code are available on OneDrive. You can download them using the following link (it requires access):

[south west leak]([https://onedrive.live.com/link-to-your-folder](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EaBI3gScakBIqTOhfaaOB-4BiFrfXQjREFYca4D2LvStHw?e=SDvZUI))
[west leak]([[https://onedrive.live.com/link-to-your-folder](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EaBI3gScakBIqTOhfaaOB-4BiFrfXQjREFYca4D2LvStHw?e=SDvZUI)](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/Eb-OsjTDF4RNs80UD47DoBYBdHlNVM1LveSs4H9zXcghpg?e=fLDFUO))
[south leak]([[[https://onedrive.live.com/link-to-your-folder](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EaBI3gScakBIqTOhfaaOB-4BiFrfXQjREFYca4D2LvStHw?e=SDvZUI)](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/Eb-OsjTDF4RNs80UD47DoBYBdHlNVM1LveSs4H9zXcghpg?e=fLDFUO)](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/Ebc2f4NVjxxIuyxgQoz9v0gBF_qzYmek3DuHFVyt7yT9bw?e=Zhmzkj))
[north leak](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/ESofkFzl6yNDnVK7NjmONooBjVq2a43GhgDA2TV98Cy2Yg?e=3VPKNu)
[north_west leak](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/ESmA3K0eppVMj8Z7_3VhHocBJocIEksMZy_XhCDLhL-I3Q?e=Gpj1Mp)
[east leak]([https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/ESmA3K0eppVMj8Z7_3VhHocBJocIEksMZy_XhCDLhL-I3Q?e=Gpj1Mp](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EcmpnOCUHYpOsdBdcXvCCf4Bdoc5lz_JPk66X-zBaU9jfw?e=4hgm8f))
[east leak](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EcmpnOCUHYpOsdBdcXvCCf4Bdoc5lz_JPk66X-zBaU9jfw?e=4hgm8f)
[single](https://ostatemailokstate-my.sharepoint.com/:u:/g/personal/mehrdad_zomorodiyan_okstate_edu/EUgYf9oKTHlAn11Pv5NGiPwBeQrPba1jQXJQBi-MZSrSpw?e=9qchbe)

After downloading, place the files in the `data` directory of the repository.
