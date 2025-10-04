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
sp_ymin = 1e-14
sp_ymax = 5e-4

sp_vmin = 1e5
sp_vmax = 1e7

case_dir = os.getenv('CASE_DIR','./output/')
data_dir = os.path.join(case_dir,'data/')
plot_dir = os.path.join(case_dir,'plots/')

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
    #plt.loglog( wavenumber, Erb, '.', label="Boundary" )
    plt.title("Rotational Spectra")
    plt.xlabel("\sqrt{\lambda} (1/m)")
    plt.ylabel("E ($m^2 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"rotational_spectra.eps")
    plt.close()

    print("===================================")
    print("   Divergent modes")
    print("===================================")
    e_d, Edi, Edb =  collapse_spectra( model.eval_n, spectra['divergence']['E_interior'], spectra['divergence']['E_boundary'], rtol=rtol, atol=atol)

    wavenumber = np.sqrt(e_d)
    plt.figure
    # neumann mode - divergent component
    plt.loglog( wavenumber, Edi, '.', label="Interior" )
    plt.loglog( wavenumber, Edb, '.', label="Boundary" )
    plt.title("Divergent Spectra")
    plt.xlabel("\sqrt{\sigma} (1/m)")
    plt.ylabel("E ($m^2 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"divergent_spectra.eps")
    plt.close()

if __name__ == "__main__":

    psi_mask = np.load(os.path.join(data_dir, 'psi_mask.npy'))
    q_mask = np.load(os.path.join(data_dir, 'q_mask.npy'))

    # Get the list of psi_*.npy files in the current directory that are not psi_mask.npy or q_mask.npy
    if not os.path.exists(data_dir):
        print(f"Case directory {data_dir} does not exist.")
        sys.exit(1)

    files = [f for f in os.listdir(data_dir) if f.startswith('psi_') and f.endswith('.npy') and f not in ['psi_mask.npy']]
    if not files:
        print("No psi_*.npy files found in the current directory.")
        sys.exit(1)

    # Load parameters
    param = load_param(data_dir)
    param['device'] = device
    param['dtype'] = dtype
    dx = param['Lx'] / param['nx']
    dy = param['Ly'] / param['ny']
    area = psi_mask.sum()*dx*dy
    print(f"Area of the domain: {area:.6e} m^2")

    nma_obj = NMA(param,model=TUML)
    nma_obj.load(data_dir)

    print(f"Device: {nma_obj.device}")
    print(f"Data type: {nma_obj.dtype}")

    nma_obj.plot_eigenmodes()

    spectra_output_file = f"{data_dir}/spectra.npz"

    if not os.path.exists(f"{data_dir}/spectra.npz") or force_recompute :

        # Load the MQGeometry stream function from .npy output
        tmp = np.load(os.path.join(data_dir,files[0]))
        psi = np.empty((len(files),) + tmp.shape)
        for i,f in enumerate(files):
            print(f"Loading stream function from {f}")
            psi[i] = np.load(os.path.join(data_dir, f))
        
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
                Eri = np.mean(data['Eri'],axis=1)/area
                Erb = np.mean(data['Erb'],axis=1)/area
                Edi = np.mean(data['Edi'],axis=1)/area
                Edb = np.mean(data['Edb'],axis=1)/area

        e_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb},
                    "divergence": {"E_interior": Edi, "E_boundary": Edb}}

    plot_spectra(nma_obj,e_spectra)
