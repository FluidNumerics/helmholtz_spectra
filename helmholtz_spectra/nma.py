

import numpy as np
import torch
from helmholtz_spectra.splig import splig, splig_load
from helmholtz_spectra.tuml import TUML
import h5py
import pickle
import os

def norm(u,area):
    """Calculates the magnitude of grid data"""
    return  torch.sqrt( torch.sum(u*u*area) )

def dot(u,v,area):
    """Performs dot product on grid data"""
    return torch.sum( u*v*area )

def load_param(case_directory):

    if not os.path.exists(f"{case_directory}/param.pkl"):
        print(f"Error : parameters file {case_directory}/param.pkl not found!")
        return None

    # Load the parameters
    with open(f'{case_directory}/param.pkl', 'rb') as f:
        return pickle.load(f)

def collapse_spectra(evals, Ei, Eb, rtol=1e-5, atol=1e-21):
    print(f" > Number of eigenvalues : {len(evals)}")

    # Ensure inputs are numpy arrays
    evals = np.asarray(evals)
    Ei = np.asarray(Ei)
    Eb = np.asarray(Eb)

    # Sort eigenvalues and associated Ei/Eb
    sort_idx = np.argsort(evals)
    e_sorted = evals[sort_idx]
    Ei_sorted = Ei[sort_idx]
    Eb_sorted = Eb[sort_idx]

    # Collapse eigenvalues
    groups = []
    group_starts = [0]
    n = len(e_sorted)
    i = 0
    while i < n:
        tol = atol + rtol * e_sorted[i]
        j = i + 1
        while j < n and np.abs(e_sorted[j] - e_sorted[i]) <= tol:
            j += 1
        group_starts.append(j)
        groups.append((i, j))
        i = j

    # Aggregate collapsed eigenvalues and energy arrays
    evals_u = []
    Ei_u = []
    Eb_u = []

    for start, end in groups:
        evals_u.append(np.mean(e_sorted[start:end]))
        Ei_u.append(np.sum(Ei_sorted[start:end]))
        Eb_u.append(np.sum(Eb_sorted[start:end]))

    print(f" > Number of collapsed eigenvalues : {len(evals_u)}")
    return np.array(evals_u), np.array(Ei_u), np.array(Eb_u)

class NMA:
    """Normal Mode Analysis class"""
    def __init__(self,param,model=TUML):
        self.initialized = True
        self.param = param

        self.splig_d = None
        self.evec_d = None
        self.eval_d = None

        self.splig_n = None
        self.evec_n = None
        self.eval_n = None
        self.device = param['device']
        self.dtype = param['dtype']
        self.arr_kwargs = {'dtype':self.dtype, 'device': self.device}

        self.case_directory = param['case_directory']

        # Set the inner_product and norm definitions
        self.inner_product = dot
        self.norm = norm

        # Initialize the model
        self.model = model(param)


    def construct_splig(self):
        # Getting the dirichlet mode mask, grid, and laplacian operator.
        self.mask_d = self.model.masks.psi.type(torch.int32).squeeze().cpu().numpy()
        self.mask_n = self.model.masks.q.type(torch.int32).squeeze().cpu().numpy()
        print(f"------------------------------")
        print(f"Building dirichlet mode matrix")
        print(f"------------------------------")
        self.splig_d = splig(self.mask_d,self.model.apply_laplacian_d) # Dirichlet mode 
        print(f"")
        print(f"----------------------------")
        print(f"Building neumann mode matrix")
        print(f"----------------------------")
        self.splig_n = splig(self.mask_n,self.model.apply_laplacian_n) # Neumann mode 
        print(f"")

    def write(self):

        with open(f'{self.case_directory}/param.pkl', 'wb') as f:
            pickle.dump(self.param, f)

        # Write structures to file
        filename = f"{self.case_directory}/dirichlet"
        self.splig_d.write(filename)

        filename = f"{self.case_directory}/neumann"
        self.splig_n.write(filename)



    def load(self, case_directory):

        def Filter(string, substr):
            return [str for str in string if any(sub in str for sub in substr)]

        self.case_directory = case_directory
        self.splig_d = splig_load(f"{case_directory}/dirichlet")
        self.mask_d = torch.from_numpy(self.splig_d.mask).to(self.device, dtype=self.dtype)
        if not os.path.exists(f"{case_directory}/dirichlet.evec.batched.h5"):
            print(f"Error : dirichlet.evec.batched.h5 not found in {case_directory}. Please run the reformat_hdf5.py script to generate it from the SLEPC output.")
            return None
        print(f"Loading dirichlet mode eigenvectors from {case_directory}/dirichlet.evec.batched.h5")
        self.evec_d = h5py.File(f"{case_directory}/dirichlet.evec.batched.h5",'r')["eigenmodes"]
        print(f"Loading dirichlet mode eigenvalues from {case_directory}/dirichlet.eval.h5")
        fobj = h5py.File(f"{case_directory}/dirichlet.eval.h5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_d = fobj[obj_key][:]


        self.neval_d = self.eval_d.shape[0]
        print(f"Number of dirichlet eigenmodes : {self.neval_d}")
        print("")

        self.splig_n = splig_load(f"{case_directory}/neumann")
        self.mask_n = torch.from_numpy(self.splig_n.mask).to(self.device, dtype=self.dtype)
        if not os.path.exists(f"{case_directory}/neumann.evec.batched.h5"):
            print(f"Error : neumann.evec.batched.h5 not found in {case_directory}. Please run the reformat_hdf5.py script to generate it from the SLEPC output.")
            return None
        print(f"Loading neumann mode eigenvectors from {case_directory}/neumann.evec.batched.h5")
        self.evec_n = h5py.File(f"{case_directory}/neumann.evec.batched.h5",'r')["eigenmodes"]
        print(f"Loading neumann mode eigenvalues from {case_directory}/neumann.eval.h5")
        fobj = h5py.File(f"{case_directory}/neumann.eval.h5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_n = fobj[obj_key][:]

        self.neval_n = self.eval_n.shape[0]
        print(f"Number of neumann eigenmodes : {self.neval_n}")
    
    def projection_coefficients(self,u,v,batch_size=1000):
        """
        This routine calculates the following projection coefficiens

            di_m - Divergent (Neumann) mode projection coefficients, interior component
            db_m - Dirichlet (Neumann) mode projection coefficients, boundary component
            vi_m - Vorticity (Dirichlet) mode projection coefficients, interior component
            vb_m - Vorticity (Dirichlet) mode projection coefficients, interior component
        """    
        from tqdm import tqdm

        du = self.model.divergence(u,v).squeeze() # Computes divergence field from the velocity field
        ndof = int(self.mask_n.sum().item()) # Number of degrees of freedom
        divu = torch.zeros((du.shape[0],ndof), dtype=self.dtype, device=self.device) # Initialize divergence field
        for i in range(du.shape[0]):
            divu[i] = torch.masked_select(du[i,:,:], self.mask_n == 1).to(self.device, dtype=self.dtype)

        flat_area = torch.masked_select( self.model.area_n, self.mask_n == 1 ).to(self.device, dtype=self.dtype)
        weighted_divu = divu*flat_area[None,:] # Weight the divergence field by the areashape [nT,n]

        db_m = torch.zeros(
            (self.neval_n,du.shape[0]), dtype=self.dtype, device=self.device
        )  # Projection of divergence onto the neumann modes (boundary)
        di_m = torch.zeros(
            (self.neval_n,du.shape[0]), dtype=self.dtype, device=self.device
        )  # Projection of divergence onto the neumann modes (interior)

        for kstart in tqdm(range(0, self.neval_n, batch_size)):
            kend = min(kstart + batch_size, self.neval_n)
            g = torch.from_numpy(self.evec_n[kstart:kend,:]).to(dtype=self.dtype, device=self.device) 
            gmag = torch.sqrt(torch.sum(g*g*flat_area, dim=1)) # Compute norm, shape: [batch_size,1]
            ek = g/gmag[:,None] # Normalize and broadcase divide, shape: [batch_size,n]
            di_m[kstart:kend,:] = torch.matmul(ek,weighted_divu.T) # Compute projection coefficient for each mode : shape [batch_size,nT] = [batch_size,n] * [n,nT]
  
        du = self.model.vorticity(u,v).squeeze() # Computes vorticity field from the velocity field
        ndof = int(self.mask_d.sum().item()) # Number of degrees of freedom
        vort = torch.zeros((du.shape[0],ndof), dtype=self.dtype, device=self.device) # Initialize vorticity field
        for i in range(du.shape[0]):
            vort[i] = torch.masked_select(du[i,:,:], self.mask_d == 1).to(self.device, dtype=self.dtype)

        flat_area = torch.masked_select( self.model.area_d, self.mask_d == 1).to(self.device, dtype=self.dtype)
        weighted_vort = vort*flat_area[None,:] # shape [nT,n]

        vb_m = torch.zeros(
            (self.neval_d,du.shape[0]), dtype=self.dtype, device=self.device
        ) # Projection of vorticity onto the dirichlet modes (boundary)
        vi_m = torch.zeros(
            (self.neval_d,du.shape[0]), dtype=self.dtype, device=self.device
        )  # Projection of vorticity onto the dirichlet modes (interior)
        
        for kstart in tqdm(range(0, self.neval_d, batch_size)):
            kend = min(kstart + batch_size, self.neval_d)
            print(f"Computing projection coefficients for dirichlet modes {kstart} to {kend} of {self.neval_d}")
            g = torch.from_numpy(self.evec_d[kstart:kend,:]).to(dtype=self.dtype, device=self.device)
            gmag = torch.sqrt(torch.sum(g*g*flat_area, dim=1))  # Compute norm, shape: [batch_size,1]
            print(f"gmag shape: {gmag.shape}")
            print(f"g shape: {g.shape}")
            ek = g/gmag[:,None] # Normalize and broadcase divide, shape: [batch_size,n]
            print(f"ek shape: {ek.shape}")
            print(f"weighted_vort shape: {weighted_vort.shape}")
            vi_m[kstart:kend,:] = torch.matmul(ek,weighted_vort.T) # Compute projection coefficient for each mode : shape [batch_size,nT] = [batch_size,n] * [n,nT]

        return vi_m.cpu().numpy(), vb_m.cpu().numpy(), di_m.cpu().numpy(), db_m.cpu().numpy()

    def spectra(self, u, v, spectra_output_file, rtol=1e-5, atol=1e-21,batch_size=1000):
        """Calculates the energy spectra for a velocity field (u,v).

        The velocity field components are assumed to be on the u and v points of an arakawa c-grid.
        
        The energy is broken down into four parts

            1. Divergent interior
            2. Rotational interior
            3. Divergent boundary
            4. Rotational boundary
        
        Each component is defined as

            1. Edi_{m} = -0.5*di_m*di_m/\lambda_m 
            2. Eri_{m} = -0.5*vi_m*vi_m/\sigma_m 
            3. Edb_{m} = -(0.5*db_m*db_m + db_m*di_m)/\lambda_m 
            4. Erb_{m} = -(0.5*vb_m*vb_m + vb_m*vi_m)/\sigma_m         

        Once calculated, the spectra is constructed as four components

            1. { \lambda_m, Edi_m }_{m=0}^{N}
            2. { \sigma_m, Eri_m }_{m=0}^{N}
            3. { \lambda_m, Edb_m }_{m=0}^{N}
            4. { \sigma_m, Erb_m }_{m=0}^{N}
 
        Energy associated with degenerate eigenmodes are accumulated to a single value. Eigenmodes are deemed
        "degenerate" if their eigenvalues similar out to "decimals" decimal places. The eigenvalue chosen
        for the purpose of the spectra is the average of the eigenvalues of the degenerate modes.
        
        """
    

        # Compute the projection coefficients
        print(" > Computing projection coefficients")
        vi_m, vb_m, di_m, db_m = self.projection_coefficients(u,v,batch_size)

        print(" > Computing energy associated with interior vorticity")
        # Calculate the energy associated with interior vorticity
        n_zeros = np.zeros_like(self.eval_n)
        zero_mode_mask = np.isclose(self.eval_n, n_zeros, rtol=rtol, atol=atol)
        Edi = 0.5 * di_m * di_m / self.eval_n[:,None]
        #Edi[zero_mode_mask] = 0.0

        print(" > Computing energy associated with boundary circulation")
        # Calculate the energy associated with boundary vorticity
        Edb = (0.5 * db_m * db_m + di_m*db_m) / self.eval_n[:,None]
        #Edb[zero_mode_mask] = 0.0

        print(" > Computing energy associated with interior divergence")
        # Calculate the energy associated with interior divergence
        Eri = 0.5 * vi_m * vi_m / self.eval_d[:,None]

        print(" > Computing energy associated with boundary normal flow")
        # Calculate the energy associated with boundary divergence
        Erb = (0.5 * vb_m * vb_m + vi_m*vb_m) / self.eval_d[:,None]

        print(f" Saving energy spectra to file {spectra_output_file}")
        np.savez( spectra_output_file, Eri = Eri, Erb = Erb, vi_m = vi_m, vb_m = vb_m,
                                           Edi = Edi, Edb = Edb, di_m = di_m, db_m = db_m )

        energy_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb, 
                                        "p_interior": vi_m, "p_boundary": vb_m,
                                        "eval": self.eval_d},
                         "divergence": {"E_interior": Edi, "E_boundary": Edb, 
                                        "p_interior": di_m, "p_boundary": db_m,
                                        "eval": self.eval_n}}
        return energy_spectra

    def plot_eigenmodes(self,vscale=8e-3):
        import matplotlib.pyplot as plt
        import math
        import os
        import numpy as np
        import numpy.ma as ma

        plot_dir = f'{self.case_directory}/eigenmodes'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        niter = range(self.neval_n-50,self.neval_n)
        diter = range(self.neval_d-50,self.neval_d)

        for k in niter:
            f,a = plt.subplots(1,1)
            vflat = self.evec_n[k,:]
            v = ma.array( np.zeros((self.splig_n.nx,self.splig_n.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_n.mask-1),fill_value=0.0 )
            v[~v.mask] = vflat
            im = a.imshow(v,cmap='RdBu',vmin=-vscale,vmax=vscale)
            a.grid(None)
            f.colorbar(im, ax=a,fraction=0.046,location='right')
            a.set_title(f'e_{k}')

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/neumann_modes_{k}.png')
            plt.close()

        for k in diter:
            f,a = plt.subplots(1,1)
            vflat = self.evec_d[k,:]
            v = ma.array( np.zeros((self.splig_d.nx,self.splig_d.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_d.mask-1),fill_value=0.0 )
            v[~v.mask] = vflat            
            im = a.imshow(v,cmap='RdBu',vmin=-vscale,vmax=vscale)
            a.grid(None)
            f.colorbar(im, ax=a,fraction=0.046,location='right')
            a.set_title(f'e_{k}')

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/dirichlet_modes_{k}.png')
            plt.close()
