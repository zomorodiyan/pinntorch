import numpy as np
import torch

def preprocess_data(file_paths):
    # Read all data from files
    all_data_files = []
    num_timesteps = None

    for file_path in file_paths:
        print(f'file: {file_path}')
        data = np.load(file_path, allow_pickle=True).item()
        u_ref = np.array(data["u"].astype(float))
        v_ref = np.array(data["v"].astype(float))
        p_ref = np.array(data["p"].astype(float))
        k_ref = np.array(data["k"].astype(float))
        omega_ref = np.array(data["omega"].astype(float))
        c_ref = np.array(data["c"].astype(float))
        coords = np.array(data["coords"].astype(float))
        Re = float(data["Re"])
        theta = float(data["theta"])

        # Non-dimensionalize data
        L_star = 80.0
        U_star = 9.0
        coords = coords / L_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        p_ref = p_ref / U_star**2
        k_ref = k_ref / U_star**2
        omega_ref = omega_ref * L_star / U_star

        # Clip extreme values
        k_ref = np.clip(k_ref, 1e-8, 0.1)
        omega_ref = np.clip(omega_ref, 1e-8, 0.5) # max should be equal to omega at inlet
        c_ref = np.clip(c_ref, 1e-8, 0.00001)

        # Store processed data
        all_data_files.append((coords, u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, Re, theta))
        if num_timesteps is None:
            num_timesteps = u_ref.shape[0]

    all_data = []

    # Interleave data points from all files
    num_points = all_data_files[0][0].shape[0]
    for t in range(num_timesteps):
        if t % 10 == 0:
            print(f't: {t}')
        for p in range(num_points):
            for data in all_data_files:
                coords, u_ref, v_ref, p_ref, k_ref, omega_ref, c_ref, Re, theta = data
                all_data.append((coords[p, 0], coords[p, 1], t, Re, theta, u_ref[t, p], v_ref[t, p], p_ref[t, p], k_ref[t, p], omega_ref[t, p], c_ref[t, p]))
    return np.array(all_data)

# Preprocess data
'''
file_paths = ["data/unsteady_emh.npy", "data/unsteady_nemh.npy",
              "data/unsteady_nwmh.npy", "data/unsteady_swmh.npy",
              "data/unsteady_smh.npy"]
'''
file_paths = ["data/unsteady_emh.npy"]
data_array = preprocess_data(file_paths)

# Save preprocessed data
np.save("data/preprocessed_emh_cliped.npy", data_array)

'''
def preprocess_data(file_paths):
    all_data = []

    for file_path in file_paths:
        print(f'file: {file_path}')
        data = np.load(file_path, allow_pickle=True).item()
        u_ref = np.array(data["u"].astype(float))
        v_ref = np.array(data["v"].astype(float))
        p_ref = np.array(data["p"].astype(float))
        k_ref = np.array(data["k"].astype(float))
        omega_ref = np.array(data["omega"].astype(float))
        c_ref = np.array(data["c"].astype(float))
        coords = np.array(data["coords"].astype(float))
        Re = float(data["Re"])
        theta = float(data["theta"])

        # Non-dimensionalize data
        L_star = 80.0
        U_star = 9.0
        coords = coords / L_star
        u_ref = u_ref / U_star
        v_ref = v_ref / U_star
        p_ref = p_ref / U_star**2
        k_ref = k_ref / U_star**2
        omega_ref = omega_ref * L_star / U_star

        # Clip extreme values
        k_ref = np.clip(k_ref, 1e-8, 0.1)
        omega_ref = np.clip(omega_ref, 1e-8, 0.5)
        c_ref = np.clip(c_ref, 1e-8, 5e-5)

        # Create a large dataset combining all variables
        num_timesteps, num_points = u_ref.shape
        for t in range(num_timesteps):
            if t % 10 == 0: print(f't: {t}')
            for p in range(num_points):
                all_data.append((coords[p, 0], coords[p, 1], t, Re, theta, u_ref[t, p], v_ref[t, p], p_ref[t, p], k_ref[t, p], omega_ref[t, p], c_ref[t, p]))

    return np.array(all_data)

# Preprocess data
file_paths = ["data/unsteady_emh.npy", "data/unsteady_nemh.npy",
              "data/unsteady_nwmh.npy", "data/unsteady_swmh.npy",
              "data/unsteady_smh.npy"]
data_array = preprocess_data(file_paths)

# Save preprocessed data
np.save("data/preprocessed_data.npy", data_array)
'''

