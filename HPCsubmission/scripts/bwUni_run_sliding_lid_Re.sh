#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --mem=90000
#SBATCH --time=04:00:00
#SBATCH --partition=multiple
#SBATCH --output=run_Re_o.out
#SBATCH --error=run_Re_o.err
#SBATCH --export=ALL

module load compiler/gnu/10.2   
module load mpi/openmpi/4.1
module load devel/python/3.8.6_gnu_10.2

echo "Simulate Sliding Lid induced capacity for different Re"
echo "--------------------------------"

echo "Initiating flow pattern comparison by varying omega to simulate different Reynold's number" 
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -u 0.05 -re_min 100 -re_max 1000 -re_count 10 -nt 50000 -run_multiple omega
echo "Completed"

echo "Initiating flow pattern comparison by varying velocity to simulate different Reynold's number" 
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -o 1.2 -re_min 100 -re_max 1000 -re_count 10 -nt 50000 -run_multiple velocity
echo "Completed"
