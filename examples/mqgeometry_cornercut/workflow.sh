#!/bin/bash
# This script is used to launch the complete workflow for demonstrating spectra in irregular domains
# Here, we run an MQGeometry simulation and create sparse SLEPC matrices (sim.py) that are then passed 
# off to a SLEPC solver to compute the Dirichlet and Neumann modes of the domain.
# Once the simulation is complete and the Dirichlet and Neumann modes are computed, we then
# compute the spectra for each time level using the `helmholtz_spectra` package


###############################################################################################
# Run the MQGeometry simulation
###############################################################################################

simulation_jobid=$(sbatch --time=8:00:00 sim.sh)

echo "Started MQGeometry simulation job with ID: ${simulation_jobid##* }"

###############################################################################################
# Compute the dirichlet and neumann modes for the domain
###############################################################################################

eigenmodes_jobid=$(sbatch --time=10:00:00 eigenmodes.sh)

echo "Started eigenmodes job with ID: ${eigenmodes_jobid##* }"

###############################################################################################
# Calculate the spectra for each time level
###############################################################################################

#spectra_jobid=$(sbatch --time=1:00:00 --dependency=afterok:${simulation_jobid##* }:${eigenmodes_jobid##* } spectra.sh)

#echo "Started spectra calculation job with ID: ${spectra_jobid##* }"