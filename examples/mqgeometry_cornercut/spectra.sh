#!/bin/bash
#SBATCH -n1
#SBATCH -c16
#SBATCH --mem=100G
#SBATCH --gres=gpu:mi210:1
#SBATCH --job-name=helmholtz_spectra_spectra
#SBATCH --output=./helmholtz_spectra_spectra.out
#SBATCH --error=./helmholtz_spectra_spectra.out
#SBATCH --nodelist=noether

WORKDIR=/scratch/joe/mqgeometry_cornercut
mkdir -p $WORKDIR
cp spectra.py $WORKDIR

###############################################################################################
#   Setup the software environment
###############################################################################################
source ./galapagos_env.sh 
module list
conda env list

###############################################################################################
# Run the Spectra Calculation
###############################################################################################
cd $WORKDIR
python spectra.py 