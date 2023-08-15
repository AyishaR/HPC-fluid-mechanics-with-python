#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --mem=90000
#SBATCH --time=00:30:00
#SBATCH --partition=dev_multiple
#SBATCH --output=run_Re.out
#SBATCH --error=run_Re.err
#SBATCH --export=ALL

module load compiler/gnu/10.2   
module load mpi/openmpi/4.1
module load devel/python/3.8.6_gnu_10.2

echo "Simulate Sliding Lid induced capacity for different Re\n"
echo "--------------------------------"

echo "Initiating flow pattern comparison by varying omega to simulate different Reynold's number" 
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -u 0.05 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple omega
echo "Completed\n"

echo "Initiating flow pattern comparison by varying velocity to simulate different Reynold's number" 
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -o 1.2 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple velocity
echo "Completed\n"

# echo "Initiating flow pattern comparison by varying grid size to simulate different Reynold's number" 
# mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -u 0.05 -o 0.8 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple grid
# echo "Completed\n"