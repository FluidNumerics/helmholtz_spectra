#!/bin/bash
#SBATCH -n1
#SBATCH -c8
#SBATCH --gres=gpu:mi210:1
#SBATCH --job-name=helmholtz_spectra_sim
#SBATCH --output=./helmholtz_spectra_sim.out
#SBATCH --error=./helmholtz_spectra_sim.out
#SBATCH --nodelist=noether


###############################################################################################
#   Setup the software environment
###############################################################################################
source ./galapagos_env.sh 
module list
conda env list

###############################################################################################
# Run the MQGeometry simulation
###############################################################################################
cp $exampledir/sim.py $workdir
cd $workdir
python sim.py # Launch the mqgeometry simulation

###############################################################################################
# Copy the results back to the submit directory
###############################################################################################
cp $workdir/psi_*.npy $SLURM_SUBMIT_DIR # Copy the simulation results back to the submit directory
cp $workdir/psi_mask.npy $SLURM_SUBMIT_DIR # Copy the mask back to the submit directory
cp $workdir/q_mask.npy $SLURM_SUBMIT_DIR # Copy the mask back to the submit directory
cp $workdir/dirichlet.dat $SLURM_SUBMIT_DIR # Copy the dirichlet.dat file back to the submit directory
cp $workdir/dirichlet.dat.info $SLURM_SUBMIT_DIR # Copy the dirichlet.dat.info file back to the submit directory
cp $workdir/neumann.dat $SLURM_SUBMIT_DIR # Copy the neumann.dat file back to the submit directory
cp $workdir/neumann.dat.info $SLURM_SUBMIT_DIR # Copy the neumann.dat.info file back to the submit directory
cp $workdir/*.png $SLURM_SUBMIT_DIR # Copy any PNG files back to the submit directory
cp $workdir/param.pkl $SLURM_SUBMIT_DIR # Copy the parameters back to the submit directory
