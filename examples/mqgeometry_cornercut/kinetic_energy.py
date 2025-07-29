#!/usr/bin/env python
# 
# This example is meant to show a complete walkthrough for computing
# the dirichlet and neumann modes for the wind-driven gyre example from
# L. Thiry's MQGeometry.
#
# Once the sparse matrices are created with this script, the dirichlet
# and neumann mode eigenpairs can be diagnosed with ../bin/laplacian_modes
#
# From here, the eigenmodes and eigenvalues can be used to calcualte the spectra 
# of the velocity field obtained with a QG simulation from MQGeometry.
# 
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from helmholtz_spectra.nma import NMA, load_param, collapse_spectra
from helmholtz_spectra.tuml import TUML
import os
import sys
from torch.profiler import profile, record_function, ProfilerActivity

plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
print(f" Device: {device}, Data type: {dtype}",flush=True)
case_dir = os.getenv('CASE_DIR','./output/')
data_dir = os.path.join(case_dir,'data/')
plot_dir = os.path.join(case_dir,'plots/')

if __name__ == "__main__":

    psi_mask = np.load(os.path.join(data_dir, 'psi_mask.npy'))
    q_mask = np.load(os.path.join(data_dir, 'q_mask.npy'))

    # Get the list of psi_*.npy files in the current directory that are not psi_mask.npy or q_mask.npy
    if not os.path.exists(data_dir):
        print(f"Case data directory {data_dir} does not exist.")
        sys.exit(1)

    files = [f for f in os.listdir(data_dir) if f.startswith('psi_') and f.endswith('.npy') and f not in ['psi_mask.npy']]
    if not files:
        print("No psi_*.npy files found in the current directory.")
        sys.exit(1)


    # Create a time series plot of the total kinetic energy
    import re
    import pandas as pd

    # Path to the simulation output file
    file_path = os.path.join(case_dir,"helmholtz_spectra_sim.out")
    # Read file content
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Define the regex pattern based on the actual output line structure
    pattern = re.compile(
        r"n=(\d+),\s*t=([\d\.]+)\s*yr,\s*q:\s*([+\-Ee\d\.]+),\s*qabs:\s*([+\-Ee\d\.]+),\s*ke0:\s*([+\-Ee\d\.]+)"
    )

    # Extract data
    records = []
    for line in lines:
        match = pattern.search(line)
        if match:
            step, time, q, qabs, ke = match.groups()
            records.append({
                "time": float(time),
                "q": float(q),
                "q_abs_max": float(qabs),
                "kinetic_energy": float(ke)
            })

    # Convert to pandas DataFrame and then to xarray Dataset
    df = pd.DataFrame(records).set_index("time")
    # Plot kinetic energy against time in this dataframe
    df['kinetic_energy'].plot(label='Kinetic Energy')
    plt.xlabel('Time (years)')
    plt.ylabel('m^4/s^2')
    plt.title('Total Kinetic Energy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'kinetic_energy_over_time.svg'))
    plt.close()


    # # Load parameters
    # param = load_param(data_dir)
    # param['device'] = device
    # param['dtype'] = dtype
    # dx = param['Lx'] / param['nx']
    # dy = param['Ly'] / param['ny']
    # area = psi_mask.sum()*dx*dy
    # print(f"Area of the domain: {area:.6e} m^2")

    # nma_obj = NMA(param,model=TUML)
    # nma_obj.load(data_dir)

    # print(f"Device: {nma_obj.device}")
    # print(f"Data type: {nma_obj.dtype}")

    

    # # Load the MQGeometry stream function from .npy output
    # tmp = np.load(os.path.join(data_dir,files[0]))
    # psi = np.empty((len(files),) + tmp.shape)
    # for i,f in enumerate(files):
    #     print(f"Loading stream function from {f}")
    #     psi[i] = np.load(os.path.join(data_dir, f))
    
    # nma_obj.model.psi = torch.from_numpy(psi).to(nma_obj.device, dtype=nma_obj.dtype)
    # u, v = nma_obj.model.get_uv() # Gets velocity field from the stream function across all time levels and layers
    # u = u[:,0,0,:,:].squeeze() # Grab surface layer and no ensemble dimension
    # v = v[:,0,0,:,:].squeeze() # Grab surface layer and no ensemble dimension

    # xv = torch.linspace(0, param['Lx'], param['nx']+1, dtype=torch.float64, device=device)
    # yv = torch.linspace(0, param['Ly'], param['ny']+1, dtype=torch.float64, device=device)
    # xc = 0.5 * (xv[1:] + xv[:-1]) # cell centers
    # yc = 0.5 * (yv[1:] + yv[:-1]) # cell centers

    # import xarray as xarray
    # import xgcm
    # print(xc.size(), yc.size(), xv[:-1].size(), yv[:-1].size())
    # print(u.shape, v.shape)
    # ds = xarray.Dataset(
    #     data_vars={
    #         'U': (['time', 'yc', 'xv'], u[:,:-1,:].cpu().numpy()),
    #         'V': (['time', 'yv', 'xc'], v[:,:,:-1].cpu().numpy())
    #     },
    #     coords={
    #         'xc': (['xc'], xc.cpu().numpy()),
    #         'xv': (['xv'], xv[:-1].cpu().numpy()),
    #         'yc': (['yc'], yc.cpu().numpy()),
    #         'yv': (['yv'], yv[:-1].cpu().numpy()),
    #         'time': (['time'], np.arange(u.shape[0]))
    #     },

    # )

    # grid = xgcm.Grid(ds, coords={
    #     'X': {'center': 'xc', 'left': 'xv'},
    #     'Y': {'center': 'yc', 'left': 'yv'},
    #     'T': {'center': 'time'}
    # })
    # ut = grid.interp(ds['U'], axis='X')
    # vt = grid.interp(ds['V'], axis='Y')
    # ke = 0.5 * (ut**2 + vt**2)
    # for i,f in enumerate(files):
    #     iterate = int(f.split('_')[1].split('.')[0][:-1])
    #     t=iterate*param['dt']
    #     ke_plot = ke[i,:,:].transpose('xc', 'yc').compute()
    #     ke_plot.plot(vmin=0, vmax=0.5, cmap='binary')
    #     plt.title(f'Kinetic Energy, {t/(365*86400):.2f} yrs')
    #     plt.xlabel('x(m)')
    #     plt.ylabel('y(m)')
    #     plt.savefig(os.path.join(plot_dir, f'{f}.ke.png'))
    #     plt.close()
    #     print(ds)
    #     print(grid)