#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --mem=90000
#SBATCH --time=00:30:00
#SBATCH --partition=dev_multiple
#SBATCH --output=run_sliding_lid.out
#SBATCH --error=run_sliding_lid.err
#SBATCH --export=ALL

module load compiler/gnu/10.2   
module load mpi/openmpi/4.1
module load devel/python/3.8.6_gnu_10.2

echo "Sliding Lid simulation (in parallel)..."
echo "--------------------------------"
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -o 1.2 -nx 300 -ny 300 -nt 50000 -nt_log 5000
echo "Completed"
