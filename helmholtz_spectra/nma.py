

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

def collapse(e,rtol,atol):
    """Collapses elements of e that are close to each other within specified tolerance"""
    
    n = e.shape[0]
    i = 0
    e_collapsed = np.zeros_like(e)
    while i < n:
        k = []
        for j in range(i,n):
            if np.abs(e[i]-e[j]) <= atol + rtol*e[i]:
                k.append(i)
            else:
                break
        eavg = np.mean(e[k])
        e_collapsed[k] = eavg
        i = k[-1]+1
        
    return e_collapsed

def collapse_spectra( evals, Ei, Eb, rtol=1e-5, atol=1e-21 ):
        
        print(f" > Number of eigenvalues : {len(evals)}")
        evals_c = collapse(evals,rtol,atol)
        evals_u = np.unique(evals_c)
        print(f" > Number of collapsed eigenvalues : {len(evals_u)}")

        Ei_u = np.zeros_like(evals_u)
        Eb_u = np.zeros_like(evals_u)
        k = 0
        for ev in evals_u:
            Ei_u[k] = np.sum(Ei[evals_c == ev])
            Eb_u[k] = np.sum(Eb[evals_c == ev])
            k+=1

        return evals_u, Ei_u, Eb_u

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
        self.mask_d = torch.from_numpy(self.splig_d.mask)
        print(f"Loading dirichlet mode eigenvectors from {case_directory}/dirichlet.evec.h5")
        self.evec_d = h5py.File(f"{case_directory}/dirichlet.evec.h5",'r')
        print(f"Loading dirichlet mode eigenvalues from {case_directory}/dirichlet.eval.h5")
        fobj = h5py.File(f"{case_directory}/dirichlet.eval.h5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_d = fobj[obj_key][:]

        # Get the keys for only the real components of the eigenvectors
        self.evec_d_rkeys = Filter(self.evec_d.keys(),['Xr'])
        self.evec_d_tag = "_".join(self.evec_d_rkeys[0].split("_")[1:])

        self.neval_d = int(len(self.evec_d_rkeys))
        print(f"Number of dirichlet eigenmodes : {self.neval_d}")
        print("")

        self.splig_n = splig_load(f"{case_directory}/neumann")
        self.mask_n = torch.from_numpy(self.splig_n.mask)
        print(f"Loading neumann mode eigenvectors from {case_directory}/neumann.evec.h5")
        self.evec_n = h5py.File(f"{case_directory}/neumann.evec.h5",'r')
        print(f"Loading neumann mode eigenvalues from {case_directory}/neumann.eval.h5")
        fobj = h5py.File(f"{case_directory}/neumann.eval.h5",'r')
        obj_key = Filter(fobj.keys(),['eigr'])[0]
        self.eval_n = fobj[obj_key][:]

        # Get the keys for only the real components of the eigenvectors
        self.evec_n_rkeys = Filter(self.evec_n.keys(),['Xr'])
        self.evec_n_tag = "_".join(self.evec_n_rkeys[0].split("_")[1:])

        self.neval_n = int(len(self.evec_n_rkeys))
        print(f"Number of neumann eigenmodes : {self.neval_n}")

        filename = f"{self.case_directory}/eval-xydecomp.npz"
        if os.path.exists(filename):
            with np.load(filename) as data:
                self.n_x = data['n_x']
                self.n_y = data['n_y']
                self.d_x = data['d_x']
                self.d_y = data['d_y']
        else:
            print(" > Computing eigenvalue xy-decomposition")
            self.n_x, self.n_y, self.d_x, self.d_y = self.eigenmode_xydecomp()
            np.savez( filename, n_x = self.n_x, n_y = self.n_y, d_x = self.d_x, d_y = self.d_y )

    def get_dirichlet_mode(self,k):
        import numpy as np
        import numpy.ma as ma

        if k < self.neval_d:
            obj_key = f"Xr{k}_{self.evec_d_tag}"
            if obj_key in list(self.evec_d_rkeys):
                v = self.evec_d[obj_key]
                v_gridded = ma.array( np.zeros((self.splig_d.nx,self.splig_d.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_d.mask-1),fill_value=0.0 )
                v_gridded[~v_gridded.mask] = v
                return v_gridded
            else: 
                print(f"{obj_key} not found in dirichlet eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of dirichlet modes {self.neval_d}")
            return None

    def get_neumann_mode(self,k):
        import numpy as np
        import numpy.ma as ma

        if k < self.neval_n:
            obj_key = f"Xr{k}_{self.evec_n_tag}"
            if obj_key in list(self.evec_n_rkeys):
                v = self.evec_n[obj_key]
                v_gridded = ma.array( np.zeros((self.splig_n.nx,self.splig_n.ny)), dtype=np.float64, order='C', mask=np.abs(self.splig_n.mask-1),fill_value=0.0 )
                v_gridded[~v_gridded.mask] = v
                return v_gridded
            else: 
                print(f"{obj_key} not found in neumann eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of neumann modes {self.neval_n}")
            return None

    def get_flat_dirichlet_mode(self,k):
        """ Returns a torch tensor for the k-th dirichlet mode. A 1-D flat array consisting of data only at the interior points is returned. """

        if k < self.neval_d:
            obj_key = f"Xr{k}_{self.evec_d_tag}"
            if obj_key in list(self.evec_d_rkeys):
                return torch.from_numpy(self.evec_d[obj_key][:])
            else: 
                print(f"{obj_key} not found in dirichlet eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of dirichlet modes {self.neval_d}")
            return None

    def get_flat_neumann_mode(self,k):

        if k < self.neval_n:
            obj_key = f"Xr{k}_{self.evec_n_tag}"
            if obj_key in list(self.evec_n_rkeys):
                return torch.from_numpy(self.evec_n[obj_key][:])
            else: 
                print(f"{obj_key} not found in neumann eigenvectors h5 index.")
        else:
            print(f"{k} exceeds number of neumann modes {self.neval_n}")
            return None

    def projection_coefficients(self,u,v):
        """
        This routine calculates the following projection coefficiens

            di_m - Divergent (Neumann) mode projection coefficients, interior component
            db_m - Dirichlet (Neumann) mode projection coefficients, boundary component
            vi_m - Vorticity (Dirichlet) mode projection coefficients, interior component
            vb_m - Vorticity (Dirichlet) mode projection coefficients, interior component
        """

        divu = torch.masked_select( self.model.divergence(u,v), self.mask_n == 1 )
        flat_area = torch.masked_select( self.model.area_n, self.mask_n == 1)
        db_m = np.zeros(
            (self.neval_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (boundary)
        di_m = np.zeros(
            (self.neval_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (interior)

        # Interior divergence contribution
        for k in np.arange(self.neval_n):
            g = self.get_flat_neumann_mode(k)
            gmag = self.norm( g, flat_area )
            ek = g/gmag
            di_m[k] = self.inner_product(divu,ek,flat_area)  # Projection of divergence onto the neumann modes

            # Boundary divergence contribution
            g = torch.from_numpy(self.get_neumann_mode(k).data).reshape(1,self.splig_n.nx,self.splig_n.ny)
            ek = g/gmag # normalize

            # Map the neumann mode from the tracer points to u-points and v-points
            eku = self.model.map_T_to_U(ek)*u
            ekv = self.model.map_T_to_V(ek)*v

            # Compute \div( \vec{u} e_k )
            divuek = torch.masked_select( self.model.divergence(eku,ekv), self.mask_n == 1 )

            # Then we need to compute -\int( \div( \vec{u} e_k ) dA )
            db_m[k] = -torch.sum(divuek*flat_area)

        vort = torch.masked_select( self.model.vorticity(u,v), self.mask_d == 1 )
        flat_area = torch.masked_select( self.model.area_d, self.mask_d == 1)

        vb_m = np.zeros(
            (self.neval_d), dtype=np.float64
        ) # Projection of vorticity onto the dirichlet modes (boundary)
        vi_m = np.zeros(
            (self.neval_d), dtype=np.float64
        )  # Projection of vorticity onto the dirichlet modes (interior)

        for k in np.arange(self.neval_d):
            g = self.get_flat_dirichlet_mode(k)
            gmag = self.norm(g,flat_area)
            ek = g/gmag
            vi_m[k] = self.inner_product(vort,ek,flat_area) # Projection of vorticity onto the dirichlet modes

        return vi_m, vb_m, di_m, db_m

    def eigenmode_xydecomp(self):
        """
        This routine computes the x and y variability of each eigenfunction from
        the Rayleigh quotient and assigns x and y components of the eigenvectors.

        This method explicitly assumes we are working on an Arakawa C-grid with the
        Neumann modes defined at tracer points and the Dirichlet modes defined at 
        vorticity points.
        """

        # Neumann modes
        flat_area = torch.masked_select( self.model.area_n, self.mask_n == 1)
        area_t = self.model.area_n*self.mask_n
        area_u = self.model.map_T_to_U(area_t)
        area_v = self.model.map_T_to_V(area_t)

        n_x = np.zeros(
            (self.neval_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (boundary)
        n_y = np.zeros(
            (self.neval_n), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (interior)

        for k in np.arange(self.neval_n):
            
            g = torch.from_numpy(self.get_neumann_mode(k).data).reshape(1,self.splig_n.nx,self.splig_n.ny)
            gmag = self.norm( g, area_t )
            ek = g/gmag

            ek_x = self.model.dfdx_n(ek)# dfdx for the neumann modes
            n_x[k] = torch.sum( ek_x*ek_x*area_u ).cpu().numpy()  

            ek_y = self.model.dfdy_n(ek)
            n_y[k] = torch.sum( ek_y*ek_y*area_v ).cpu().numpy()  # dfdy for the neumann modes

        # Dirichlet modes
        d_x = np.zeros(
            (self.neval_d), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (boundary)
        d_y = np.zeros(
            (self.neval_d), dtype=np.float64
        )  # Projection of divergence onto the neumann modes (interior)

        area_g = self.model.area_d*self.mask_d
        for k in np.arange(self.neval_d):
            
            g = torch.from_numpy(self.get_dirichlet_mode(k).data).reshape(1,self.splig_d.nx,self.splig_d.ny)
            gmag = self.norm( g, area_g )
            ek = g/gmag

            ek_x = self.model.dfdx_d(ek)# dfdx for the dirichlet modes
            d_x[k] = torch.sum( ek_x*ek_x*area_v ).cpu().numpy()  

            ek_y = self.model.dfdy_d(ek)
            d_y[k] = torch.sum( ek_y*ek_y*area_u ).cpu().numpy()  # dfdy for the dirichlet modes
       
        return n_x, n_y, d_x, d_y

    def spectra(self, u, v, spectra_output_file = None, rtol=1e-5, atol=1e-21):
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
        vi_m, vb_m, di_m, db_m = self.projection_coefficients(u,v)

        print(" > Computing energy associated with interior vorticity")
        # Calculate the energy associated with interior vorticity
        n_zeros = np.zeros_like(self.eval_n)
        zero_mode_mask = np.isclose(self.eval_n, n_zeros, rtol=rtol, atol=atol)
        Edi = 0.5 * di_m * di_m / self.eval_n
        Edi[zero_mode_mask] = 0.0

        print(" > Computing energy associated with boundary circulation")
        # Calculate the energy associated with boundary vorticity
        Edb = (0.5 * db_m * db_m + di_m*db_m) / self.eval_n
        Edb[zero_mode_mask] = 0.0

        print(" > Computing energy associated with interior divergence")
        # Calculate the energy associated with interior divergence
        Eri = 0.5 * vi_m * vi_m / self.eval_d

        print(" > Computing energy associated with boundary normal flow")
        # Calculate the energy associated with boundary divergence
        Erb = (0.5 * vb_m * vb_m + vi_m*vb_m) / self.eval_d

        if not spectra_output_file == None:
            np.savez( spectra_output_file, Eri = Eri, Erb = Erb, vi_m = vi_m, vb_m = vb_m,
                                           Edi = Edi, Edb = Edb, di_m = di_m, db_m = db_m )

        energy_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb, 
                                        "p_interior": vi_m, "p_boundary": vb_m},
                         "divergence": {"E_interior": Edi, "E_boundary": Edb, 
                                        "p_interior": di_m, "p_boundary": db_m}}
        return energy_spectra

    def plot_eigenmodes(self,vscale=8e-3):
        import matplotlib.pyplot as plt
        import math
        import os

        plot_dir = f'{self.case_directory}/eigenmodes'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        niter = range(self.neval_n-50,self.neval_n)
        diter = range(self.neval_d-50,self.neval_d)

        for k in niter:
            f,a = plt.subplots(1,1)
            v = self.get_neumann_mode(k)
            im = a.imshow(v,cmap='RdBu',vmin=-vscale,vmax=vscale)
            a.grid(None)
            f.colorbar(im, ax=a,fraction=0.046,location='right')
            a.set_title(f'e_{k}')

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/neumann_modes_{k}.png')
            plt.close()

        for k in diter:
            f,a = plt.subplots(1,1)
            v = self.get_dirichlet_mode(k)
            im = a.imshow(v,cmap='RdBu',vmin=-vscale,vmax=vscale)
            a.grid(None)
            f.colorbar(im, ax=a,fraction=0.046,location='right')
            a.set_title(f'e_{k}')

            plt.tight_layout()
            plt.savefig(f'{plot_dir}/dirichlet_modes_{k}.png')
            plt.close()
