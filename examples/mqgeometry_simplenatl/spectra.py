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

plt.style.use('seaborn-v0_8-whitegrid')
plt.switch_backend('agg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

force_recompute = False

# Tolerances for collapsing modes with similar eigenvalues
collapse_atol = 1e-21
collapse_rtol = 1e-2

# Plot limits for spectra
sp_xmin = 5e-6
sp_xmax = 2e-3
sp_ymin = 8e-9
sp_ymax = 4e9

sp_vmin = 1e5
sp_vmax = 1e7

case_dir = os.getenv('CASE_DIR','./')

uv_iter = 0

def plot_isotropic_spectra(model,spectra,model_input):

    rtol = 1e-11
    atol = 1e-13
    # Find dirichlet eigenvalues where x and y components match
    iso_map = np.isclose( model.d_x, model.d_y, rtol=rtol, atol=atol )
    Eri = spectra['vorticity']['E_interior'][iso_map]
    Erb = spectra['vorticity']['E_boundary'][iso_map]
    e_r = model.eval_d[iso_map]
    print(f"Number of isotropic rotational modes : {e_r.shape[0]}")

    wavenumber = 2.0*np.pi*np.sqrt(e_r)
    plt.figure
    # dirichlet mode - rotational component
    plt.loglog( wavenumber, Eri, label="Interior")
    plt.loglog( wavenumber, Erb, label="Boundary" )
    plt.title("Isotropic Rotational Spectra")
    plt.xlabel("wavenumber (rad/m)")
    plt.ylabel("E ($m^4 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{model_input}.rotational_isotropic_spectra.png")
    plt.close()

    # Find neumann eigenvalues where x and y components match
    iso_map = np.isclose( model.n_x, model.n_y, rtol=rtol, atol=atol )
    Edi = spectra['divergence']['E_interior'][iso_map]
    Edb = spectra['divergence']['E_boundary'][iso_map]
    e_d = model.eval_n[iso_map]
    print(f"Number of isotropic divergent modes : {e_d.shape[0]}")


    wavenumber = 2.0*np.pi*np.sqrt(e_d)
    plt.figure
    # neumann mode - divergent component
    plt.loglog( wavenumber, Edi, label="Interior" )
    plt.loglog( wavenumber, Edb, label="Boundary" )
    plt.title("Isotropic Divergent Spectra")
    plt.xlabel("wavenumber (rad/m)")
    plt.ylabel("E ($m^4 s^{-2}$)")
    plt.axis( xmin = sp_xmin, xmax = sp_xmax, ymin = sp_ymin, ymax = sp_ymax )
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{model_input}.divergent_isotropic_spectra.png")
    plt.close()

# def plot_decomposition_error(nma_obj,case_dir):
#     """
#     Computes the eigenvalues from the Rayleigh quotient and
#     compares them with the eigenvalues returned from SLEPc
#     """

#     decomp_err = np.abs(nma_obj.d_x + nma_obj.d_y - nma_obj.eval_d)/nma_obj.eval_d
#     plt.figure
#     plt.plot( nma_obj.eval_d, decomp_err, 'k.', markersize=0.1 )
#     plt.title("Dirichlet mode decomposition relative error")
#     plt.xlabel("$\sigma$ ($m^{-2}$)")
#     plt.ylabel("Error (\%)")
#     plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
#     plt.grid(True, which="both", ls="-", color='0.65')
#     plt.tight_layout()
#     plt.savefig(f"{case_dir}/dirichlet_mode_decomp_err.png")
#     plt.close()

#     decomp_err = np.abs(nma_obj.n_x + nma_obj.n_y - nma_obj.eval_n)/nma_obj.eval_n
#     plt.figure
#     plt.plot( nma_obj.eval_n, decomp_err, 'k.', markersize=0.1 )
#     plt.title("Neumann mode decomposition relative error")
#     plt.xlabel("$\lambda$ ($m^{-2}$)")
#     plt.ylabel("Error (\%)")
#     plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
#     plt.grid(True, which="both", ls="-", color='0.65')
#     plt.tight_layout()
#     plt.savefig(f"{case_dir}/neumann_mode_decomp_err.png")
#     plt.close()

def plot_wavenumbers(nma_obj,case_dir):
    """
    Creates plots that draw a dot at each (e_x,e_y) point from the 
    eigenvalue spatial decomposition (the x and y components of the Rayleigh quotient)
    """
    plt.figure
    plt.plot( np.sqrt(nma_obj.d_x), np.sqrt(nma_obj.d_y), 'k.', markersize=0.1 )
    plt.title("Dirichlet mode eigenvalue decomposition")
    plt.xlabel("$\sqrt{\sigma_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\sigma_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.tight_layout()
    plt.savefig(f"{case_dir}/dirichlet_mode_wavenumbers.png")
    plt.close()

    plt.figure
    plt.plot( np.sqrt(nma_obj.n_x), np.sqrt(nma_obj.n_y), 'k.', markersize=0.1 )
    plt.title("Neumann mode eigenvalue decomposition")
    plt.xlabel("$\sqrt{\lambda_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\lambda_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.tight_layout()
    plt.savefig(f"{case_dir}/neumann_mode_wavenumbers.png")
    plt.close()

def plot_total_energy_budget(nma_obj,e_spectra,model_input):

    lambda_u = nma_obj.eval_n
    Edi_u = e_spectra['divergence']['E_interior']
    Edb_u = e_spectra['divergence']['E_boundary']
    sigma_u = nma_obj.eval_d
    Eri_u = e_spectra['vorticity']['E_interior']
    Erb_u = e_spectra['vorticity']['E_boundary']

    total_energy = np.sum(Edi_u) + np.sum(Edb_u) + np.sum(Eri_u) + np.sum(Erb_u)
    print(f" Estimated total energy (area integral)   : {total_energy_m4s2:.3E} (m^4/s^2)")
    print(f" Estimated total energy (spectral sum)    : {total_energy:.3E} (m^4/s^2)")
    print(f" Estimated energy density (area integral) : {total_energy_m4s2/nma_obj.model.total_area_n():.3E} (m^2/s^2)")
    print(f" Estimated energy density (spectral sum)  : {total_energy/nma_obj.model.total_area_n():.3E} (m^2/s^2)")

    energy_components = ["Rotational (interior)","Rotational (boundary)","Divergent (interior)","Divergent (boundary)"]
    energy_values = [np.sum(Eri_u),np.sum(Erb_u),np.sum(Edi_u),np.sum(Edb_u)]
    y_pos = np.arange(len(energy_components))

    xU = max(energy_values)*1.25
    fig,ax = plt.subplots(1,1)
    ax.barh(y_pos,energy_values)
    ax.set_yticks(y_pos, labels=energy_components) 
    ax.set_xlabel("Energy ($m^4 s^{-2}$)")
    plt.xlim(0, xU)
    for y in y_pos:
        if energy_values[y] < 0.25*xU:
            ax.text(energy_values[y]+0.01*xU, y, f'{energy_values[y]:.3E} ', color='black', ha='left', va='center')
        else:
            ax.text(energy_values[y]-0.01*xU, y, f'{energy_values[y]:.3E} ', color='white', ha='right', va='center')

    plt.grid(True, which="both", ls="-", color='0.65')
    plt.tight_layout()
    plt.savefig(f"{model_input}.total_energy_budget.png")
    plt.close()

def plot_2d_spectra(nma_obj,e_spectra,model_input):

    lambda_u = nma_obj.eval_n
    Edi_u = e_spectra['divergence']['E_interior']
    Edb_u = e_spectra['divergence']['E_boundary']
    sigma_u = nma_obj.eval_d
    Eri_u = e_spectra['vorticity']['E_interior']
    Erb_u = e_spectra['vorticity']['E_boundary']

    plt.figure
    plt.scatter( np.sqrt(nma_obj.d_x), np.sqrt(nma_obj.d_y), 
        c=Eri_u, marker='o', edgecolor='none', cmap='Greys', s=1.5, norm="log",
        vmin=sp_vmin, vmax=sp_vmax)
    plt.colorbar()
    plt.title("Interior Rotational (Dirichlet) Spectra")
    plt.xlabel("$\sqrt{\sigma_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\sigma_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65',alpha=0.5)
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.savefig(f"{model_input}.rotational_interior_spectra.png")
    plt.close()

    plt.figure
    plt.scatter( np.sqrt(nma_obj.d_x), np.sqrt(nma_obj.d_y), 
        c=Erb_u, marker='o', edgecolor='none', cmap='Greys', s=1.5, norm="log",
        vmin=sp_vmin, vmax=sp_vmax)
    plt.colorbar()
    plt.title("Boundary Rotational (Dirichlet) Spectra")
    plt.xlabel("$\sqrt{\sigma_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\sigma_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65',alpha=0.5)
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.savefig(f"{model_input}.rotational_boundary_spectra.png")
    plt.close()

    plt.figure
    plt.scatter( np.sqrt(nma_obj.n_x), np.sqrt(nma_obj.n_y), 
        c=Edi_u, marker='o', edgecolor='none', cmap='Greys', s=1.5, norm="log",
        vmin=sp_vmin, vmax=sp_vmax)
    plt.colorbar()
    plt.title("Interior Divergence (Neumann) Spectra")
    plt.xlabel("$\sqrt{\lambda_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\lambda_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65',alpha=0.5)
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.savefig(f"{model_input}.divergence_interior_spectra.png")
    plt.close()

    plt.figure
    plt.scatter( np.sqrt(nma_obj.n_x), np.sqrt(nma_obj.n_y), 
        c=Edb_u, marker='o', edgecolor='none', cmap='Greys', s=1.5, norm="log",
        vmin=sp_vmin, vmax=sp_vmax)
    plt.colorbar()
    plt.title("Boundary Divergence (Neumann) Spectra")
    plt.xlabel("$\sqrt{\lambda_x}$ ($m^{-1}$)")
    plt.ylabel("$\sqrt{\lambda_y}$ ($m^{-1}$)")
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.grid(True, which="both", ls="-", color='0.65',alpha=0.5)
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.savefig(f"{model_input}.divergence_boundary_spectra.png")
    plt.close()


if __name__ == "__main__":

    psi_mask = np.load(os.path.join(case_dir, 'psi_mask.npy'))
    q_mask = np.load(os.path.join(case_dir, 'q_mask.npy'))

    # Get the list of vort_*.npy files in the current directory
    files = [f for f in os.listdir(case_dir) if f.startswith('vort_') and f.endswith('.npy')]
    if not files:
        print("No vort_*.npy files found in the current directory.")
        sys.exit(1)
   

    # Load parameters
    param = load_param(case_dir)
    nma_obj = NMA(param,model=TUML)
    nma_obj.load(case_dir)

    plot_wavenumbers(nma_obj,case_dir)

    # Load the MQGeometry stream function from .npy output
    for f in files:
        psi = np.load(os.path.join(case_dir, f))
        nma_obj.model.psi = torch.from_numpy(psi).to(nma_obj.model.device, dtype=nma_obj.model.dtype)
        u, v = nma_obj.model.get_uv()
        print(f"min(u), max(u) : {torch.min(u)}, {torch.max(u)}")
        print(f"min(v), max(v) : {torch.min(v)}, {torch.max(v)}")

        spectra_output_file = f"{case_dir}/spectra_{f[:-4]}.npz"

        e_spectra = nma_obj.spectra(u,v,spectra_output_file=f"{case_dir}/spectra_{f[:-4]}.npz",)
        #nma_obj.plot_eigenmodes()

        # # lambda_m, sigma_m, Edi_m, Eri_m, Edb_m, Erb_m = nma_obj.spectra(u,v,atol=1e-5)
        if (not os.path.exists(spectra_output_file)) or force_recompute:

             print("Computing Spectra")
             e_spectra = nma_obj.spectra(u,v,spectra_output_file)


        else:
            print(f"Spectra output found from previous computation. Using file {spectra_output_file}")
            with np.load(spectra_output_file) as data:
                Eri = data['Eri']
                Erb = data['Erb']
                Edi = data['Edi']
                Edb = data['Edb']

            e_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb},
                        "divergence": {"E_interior": Edi, "E_boundary": Edb}}

        
        #plot_total_energy_budget(nma_obj,e_spectra,f[:-4])

        plot_2d_spectra(nma_obj,e_spectra,f[:-4])

        plot_isotropic_spectra(nma_obj,e_spectra,f[:-4])
