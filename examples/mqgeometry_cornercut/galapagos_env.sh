#!/bin/bash


# Define the local path to the helmholtz_spectra repository
export helmholtz_spectra=/group/tdgs/joe/helmholtz_spectra

###############################################################################################
#   Setup the software environment
###############################################################################################

module purge
module load gcc/12.4.0
module load rocm/6.3.0
module load openmpi/5.0.6
module load petsc/3.23.2 slepc/3.23.1

# Set up conda
__conda_setup="$('$HOME/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Check if environment already exists
if conda env list | grep -q "helmholtz_spectra"; then
    echo "Environment 'helmholtz_spectra' already exists. Activating it."
else
    echo "Creating environment 'helmholtz_spectra'."
    conda create -n helmholtz_spectra python=3.10 --yes
fi
conda activate helmholtz_spectra