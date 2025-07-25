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
rm -rf $workdir # Remove the work directory if it exists


###############################################################################################
# Run the MQGeometry simulation
###############################################################################################
cp $exampledir/sim.py $workdir
cd $workdir
python sim.py # Launch the mqgeometry simulation

###############################################################################################
# Copy the results back to the submit directory
###############################################################################################
cp $workdir/psi_*.npy $permanent_dir/data # Copy the simulation results back to the submit directory
cp $workdir/psi_mask.npy $permanent_dir/data # Copy the mask back to the submit directory
cp $workdir/q_mask.npy $permanent_dir/data # Copy the mask back to the submit directory
cp $workdir/dirichlet.dat $permanent_dir/data # Copy the dirichlet.dat file back to the submit directory
cp $workdir/dirichlet.dat.info $permanent_dir/data # Copy the dirichlet.dat.info file back to the submit directory
cp $workdir/neumann.dat $permanent_dir/data # Copy the neumann.dat file back to the submit directory
cp $workdir/neumann.dat.info $permanent_dir/data # Copy the neumann.dat.info file back to the submit directory
cp $workdir/*.png $permanent_dir/plots # Copy any PNG files back to the submit directory
cp $workdir/param.pkl $permanent_dir/data # Copy the parameters back to the submit directory
