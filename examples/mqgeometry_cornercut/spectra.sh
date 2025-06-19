#!/bin/bash
#SBATCH -n16
#SBATCH -c2
#SBATCH --job-name=eigenmodes
#SBATCH --output=./helmholtz_spectra
#SBATCH --error=./helmholtz_spectra.stderr
#SBATCH --nodelist=noether


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
                       -eps_view_values hdf5:./dirichlet.eval.h5 > ./slepc_dirichlet.stdout
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
                       -eps_view_values hdf5:./neumann.eval.h5 > ./slepc_neumann.stdout


###############################################################################################
# Verify that MQGeometry has finished
###############################################################################################

# Poll every second until it finishes
while kill -0 $simulation_pid 2>/dev/null; do
    echo "Process $simulation_pid still running..."
    sleep 1
done

echo "Process $simulation_pid has finished."


###############################################################################################
# Calculate the spectra for each time level
###############################################################################################