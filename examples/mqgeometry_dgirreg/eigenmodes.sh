#!/bin/bash
#SBATCH -n16
#SBATCH -c2
#SBATCH --mem=200G
#SBATCH --job-name=helmholtz_spectra_eigenmodes
#SBATCH --output=./helmholtz_spectra_eigenmodes.out
#SBATCH --error=./helmholtz_spectra_eigenmodes.out
#SBATCH --nodelist=noether


###############################################################################################
#   Setup the software environment
###############################################################################################
source ./galapagos_env.sh  
module list
conda env list

###############################################################################################
# Compute the dirichlet and neumann modes for the domain
###############################################################################################
cd $workdir
# Wait until file exists
filepath=./dirichlet.dat
while [ ! -f "$filepath" ]; do
    echo "Waiting for $filepath to appear..."
    sleep 30
done
echo "$filepath is now present."
mpiexec -n ${SLURM_NTASKS} ${helmholtz_spectra}/bin/laplacian_modes -f ./dirichlet.dat \
                       -memory_view \
                       -eps_type elpa \
                       -eps_view_vectors hdf5:./dirichlet.evec.h5 \
                       -eps_view_values hdf5:./dirichlet.eval.h5
filepath=./neumann.dat
while [ ! -f "$filepath" ]; do
    echo "Waiting for $filepath to appear..."
    sleep 30
done
echo "$filepath is now present."
# Compute the neumann modes
mpiexec -n ${SLURM_NTASKS} ${helmholtz_spectra}/bin/laplacian_modes -f ./neumann.dat \
                       -memory_view \
                       -eps_type elpa \
                       -eps_view_vectors hdf5:./neumann.evec.h5 \
                       -eps_view_values hdf5:./neumann.eval.h5

###############################################################################################
# Reformat the hdf5 files to be batched
###############################################################################################
cp $exampledir/reformat_hdf5.py $workdir
cd $workdir
python reformat_hdf5.py


###############################################################################################
# Copy the results back to the submit directory
###############################################################################################
cp $workdir/dirichlet.evec.batched.h5 $SLURM_SUBMIT_DIR # Copy the dirichlet modes back to the submit directory
cp $workdir/dirichlet.eval.h5 $SLURM_SUBMIT_DIR # Copy the dirichlet eigenvalues back to the submit directory
cp $workdir/neumann.evec.batched.h5 $SLURM_SUBMIT_DIR # Copy the neumann modes back to the submit directory
cp $workdir/neumann.eval.h5 $SLURM_SUBMIT_DIR # Copy the neumann eigenvalues back to the submit directory