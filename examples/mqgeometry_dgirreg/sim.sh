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

python sim.py # Launch the mqgeometry simulation