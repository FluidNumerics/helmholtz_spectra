# MQGeometry Cornercut example

## Running the workflow

### Workflow overview
The provided `workflow.sh` file in this directory is used to submit three jobs with the appropriate dependencies
* `sim.sh` - Runs the MQGeometry simulation and creates the sparse matrices of the Dirichlet and Neumann mode Laplacian operators that can be used in SLEPC/PETSc (GPU acceleration helps produce this output quickly)
* `eigenmodes.sh` - Runs the SLEPC ELPA solver to get complete eigenmode decomposition of the Dirichlet and Neumann mode matrices. An additional post-processor is run on the `dirichlet.evec.h5` and `neumann.evec.h5` files to produce `dirichlet.evec.batched.h5` and `neumann.evec.batched.h5` files. The `.batched.h5` files consolidate the eigenvectors into a single 2-D dataset within the HDF5 file, which improves file IO performance in spectra calculation.
* `spectra.sh` - Loads all `psi_*.npy` files to compute spectra across all available time levels. Output is stored in `spectra.npz`. Additionally, a plot of the time averaged rotational energy spectra is produced as a png. GPU acceleration is highly encouraged here - computing complete spectra of 1000 time levels takes 4-5 minutes (main bottleneck is file IO)


### Running it yourself
The scripts included with this example assume you are running on Fluid Numerics' Galapagos cluster. All environment settings are managed in `./galapagos-env.sh`. This script is sourced at the beginning of each step in the workflow. Note that, if you don't have a conda environment called `helmholtz_spectra`, one will be created for you and all necessary packages will be installed. The main changes you need to make are to set the `helmholtz_spectra` and `workdir` environment variables. Default values are shown below

```
export helmholtz_spectra=/group/tdgs/$(whoami)/helmholtz_spectra # Path to the `helmholtz_spectra` repository
export workdir=/scratch/$(whoami)/mqgeometry_cornercut # Path where you want all steps to run out of
```

On Galapagos, all steps are configured to run on `noether` and the `workdir` is set to a subdirectory under `/scratch`, which is the NVMe drives on `noether`. 

To run this example, simply run `./workflow.sh` and all three steps in the calculation will be submitted to the job queue with the correct job dependencies.

Note that all data will be copied back to the directory in which you run the `./workflow.sh` script (likely this one).

## Files produced

### Simulation

When running the `sim.sh` job, the following outputs are produced : 
* `psi_*.png` and `vort_*.png` - lat/lon plots of the stream function and vorticity fields (respectively) in the upper layer of the MQGeometry simulation
* `psi_*.npy` - Stream function for all layers at each time level. Can be loaded with `np.load()`
* `dirichlet-mask.png` - Quick n' dirty lat/lon plot of the mask on the vorticity points of the grid
* `neumann-mask.png` - Quick n' dirty lat/lon plot of the mask on the stream-function points of the grid
* `dirichlet.dat` and `dirichlet.dat.info` - Sparse matrix files in PETSc/SLEPc format. These are fed into the SLEPc/ELPA program to compute the eigenmodes
* `neumann.dat` and `neumann.dat.info` - Sparse matrix files in PETSc/SLEPc format. These are fed into the SLEPc/ELPA program to compute the eigenmodes
* `dirichlet.npz` and `neumann.npz` -  
* `param.pkl` - PKL file that stores the model parameters as a python dictionary. This allows us to float the model parameters between the simulation and spectra calculation steps easily.
* `helmholtz_spectra_sim.out` - STDOUT/STDERR output file

### Eigenmodes
When running the `eigenmodes.sh` job, the following outputs are produced:
* `neumann.eval.h5` and `dirichlet.eval.h5` - Eigenvalues for the Neumann and Dirichlet modes respectively
* `neumann.evec.h5` and `dirichlet.evec.h5`  - (temporary output) Eigenvectors returned directly by SLEPC. These files are post-processed by `reformat_hdf5.py` to improve the file IO performance in `spectra.py`
* `neumann.evec.batched.h5` and `dirichlet.evec.batched.h5`  - Eigenvectors returned by `reformat_hdf5.py`. These files consolidate the eigenvectors into a single 2D dataset within each HDF5 file to provide sequential read access on the eigenvectors when computing spectra.
* `helmholtz_spectra_eigenmodes.out` - STDOUT/STDERR output file

### Spectra 
When running the `spectra.sh` job, the following outputs are produced:
* `spectra.npz` - Numpy file that contains the projection coefficients and energy spectra for all reported time levels of the MQGeometry simulation. This file can be loaded with `np.load()`, and the contents are as follows
```
energy_spectra = {"vorticity": {"E_interior": Eri, "E_boundary": Erb, 
                                        "p_interior": vi_m, "p_boundary": vb_m,
                                        "eval": self.eval_d},
                         "divergence": {"E_interior": Edi, "E_boundary": Edb, 
                                        "p_interior": di_m, "p_boundary": db_m,
                                        "eval": self.eval_n}}
```
* `trace.json` - Pytorch trace profile of the spectra calculation. Can be
* `helmholtz_spectra_spectra.out` - STDOUT/STDERR output file


###