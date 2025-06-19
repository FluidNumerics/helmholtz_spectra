#!/bin/bash
#SBATCH -n16
#SBATCH -c2
#SBATCH --job-name=helmholtz_spectra_eigenmodes
#SBATCH --output=./helmholtz_spectra_eigenmodes.out
#SBATCH --error=./helmholtz_spectra_eigenmodes.out
#SBATCH --nodelist=noether

# Define the local path to the helmholtz_spectra repository
helmholtz_spectra=/group/tdgs/joe/helmholtz_spectra

###############################################################################################
#   Setup the software environment
###############################################################################################
./galapagos_env.sh 


###############################################################################################
# Compute the dirichlet and neumann modes for the domain
###############################################################################################

# Wait until file exists
filepath=./dirichlet.dat
while [ ! -f "$filepath" ]; do
    echo "Waiting for $filepath to appear..."
    sleep 1
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
    sleep 1
done
echo "$filepath is now present."
# Compute the neumann modes
mpiexec -n ${SLURM_NTASKS} ${helmholtz_spectra}/bin/laplacian_modes -f ./neumann.dat \
                       -memory_view \
                       -eps_type elpa \
                       -eps_view_vectors hdf5:./neumann.evec.h5 \
                       -eps_view_values hdf5:./neumann.eval.h5
