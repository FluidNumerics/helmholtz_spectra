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

force_recompute = False

# Tolerances for collapsing modes with similar eigenvalues
collapse_atol = 1e-21
collapse_rtol = 5e-3

# Plot limits for spectra
sp_xmin = 5e-6
sp_xmax = 2e-3
sp_ymin = 8e-3
sp_ymax = 5e10

sp_vmin = 1e5
sp_vmax = 1e7

case_dir = os.getenv('CASE_DIR','./')

uv_iter = 0

def plot_spectra(model,spectra):

    rtol=1.0e-2
    atol=1.0e-21
    print("===================================")
    print("   Rotational modes")
    print("===================================")
    e_r, Eri, Erb =  collapse_spectra( model.eval_d, spectra['vorticity']['E_interior'], spectra['vorticity']['E_boundary'], rtol=rtol, atol=atol)

    wavenumber = 2.0*np.pi*np.sqrt(e_r)
    plt.figure
    # dirichlet mode - rotational component
    plt.loglog( wavenumber, Eri, '.', label="Interior")
    plt.loglog( wavenumber, Erb, '.', label="Boundary" )
    plt.title("Rotational Spectra")
    plt.xlabel("wavenumber (rad/m)")
    plt.ylabel("E ($m^4 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"rotational_spectra.png")
    plt.close()

    print("===================================")
    print("   Divergent modes")
    print("===================================")
    e_d, Edi, Edb =  collapse_spectra( model.eval_n, spectra['divergence']['E_interior'], spectra['divergence']['E_boundary'], rtol=rtol, atol=atol)

    wavenumber = 2.0*np.pi*np.sqrt(e_d)
    plt.figure
    # neumann mode - divergent component
    plt.loglog( wavenumber, Edi, '.', label="Interior" )
    plt.loglog( wavenumber, Edb, '.', label="Boundary" )
    plt.title("Divergent Spectra")
    plt.xlabel("wavenumber (rad/m)")
    plt.ylabel("E ($m^4 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"divergent_spectra.png")
    plt.close()

if __name__ == "__main__":

    psi_mask = np.load(os.path.join(case_dir, 'psi_mask.npy'))
    q_mask = np.load(os.path.join(case_dir, 'q_mask.npy'))

    # Get the list of psi_*.npy files in the current directory that are not psi_mask.npy or q_mask.npy
    if not os.path.exists(case_dir):
        print(f"Case directory {case_dir} does not exist.")
        sys.exit(1)

    files = [f for f in os.listdir(case_dir) if f.startswith('psi_') and f.endswith('.npy') and f not in ['psi_mask.npy']]
    if not files:
        print("No psi_*.npy files found in the current directory.")
        sys.exit(1)

    # Load parameters
    param = load_param(case_dir)
    param['device'] = device
    param['dtype'] = dtype
    nma_obj = NMA(param,model=TUML)
    nma_obj.load(case_dir)

    print(f"Device: {nma_obj.device}")
    print(f"Data type: {nma_obj.dtype}")

    nma_obj.plot_eigenmodes()

    spectra_output_file = f"{case_dir}/spectra.npz"

    if not os.path.exists(f"{case_dir}/spectra.npz") or force_recompute :

        # Load the MQGeometry stream function from .npy output
        tmp = np.load(os.path.join(case_dir,files[0]))
        psi = np.empty((len(files),) + tmp.shape)
        for i,f in enumerate(files):
            print(f"Loading stream function from {f}")
            psi[i] = np.load(os.path.join(case_dir, f))
        
        nma_obj.model.psi = torch.from_numpy(psi).to(nma_obj.device, dtype=nma_obj.dtype)
        u, v = nma_obj.model.get_uv() # Gets velocity field from the stream function across all time levels and layers
        u = u[:,0,0,:,:].squeeze() # Grab surface layer and no ensemble dimension
        v = v[:,0,0,:,:].squeeze() # Grab surface layer and no ensemble dimension

        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
            e_spectra = nma_obj.spectra(u,v,spectra_output_file=spectra_output_file,batch_size=8000)
        prof.export_chrome_trace("trace.json")
    else:
        print(f"Loading spectra from {spectra_output_file}")
        with np.load(spectra_output_file) as data:
                Eri = np.mean(data['Eri'],axis=1)
                Erb = np.mean(data['Erb'],axis=1)
                Edi = np.mean(data['Edi'],axis=1)
                Edb = np.mean(data['Edb'],axis=1)

        e_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb},
                    "divergence": {"E_interior": Edi, "E_boundary": Edb}}

    plot_spectra(nma_obj,e_spectra)
