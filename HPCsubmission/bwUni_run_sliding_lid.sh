echo "Sliding Lid simulation (in parallel)..."
echo "--------------------------------"
mpiexec -n 4 python3 -m simulations.sliding_lid_parallel -o 1.2 -nx 300 -ny 300 -nt 50000 -nt_log 5000
echo "Completed\n"
