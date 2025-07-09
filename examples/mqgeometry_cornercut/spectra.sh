#!/bin/bash
#SBATCH -n1
#SBATCH -c16
#SBATCH --mem=100G
#SBATCH --gres=gpu:mi210:1
#SBATCH --job-name=helmholtz_spectra_spectra
#SBATCH --output=./helmholtz_spectra_spectra.out
#SBATCH --error=./helmholtz_spectra_spectra.out
#SBATCH --nodelist=noether

###############################################################################################
#   Setup the software environment
###############################################################################################
source ./galapagos_env.sh 
module list
conda env list

###############################################################################################
# Run the Spectra Calculation
###############################################################################################
cp $exampledir/spectra.py $workdir
cd $workdir
python spectra.py 

###############################################################################################
# Copy the results back to the submit directory
###############################################################################################
cp $workdir/spectra.npz $SLURM_SUBMIT_DIR # Copy the spectra results back to the submit directory
cp $workdir/trace.json $SLURM_SUBMIT_DIR # Copy the profiling trace back to the submit directory
