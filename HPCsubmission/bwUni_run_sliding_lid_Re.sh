

# echo "Simulate Sliding Lid induced capacity for different Re\n"
# echo "--------------------------------"

# echo "Initiating flow pattern comparison by varying omega to simulate different Reynold's number" 
# mpiexec -n 4 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -u 0.05 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple omega
# echo "Completed\n"

# echo "Initiating flow pattern comparison by varying velocity to simulate different Reynold's number" 
# mpiexec -n 4 python3 -m simulations.sliding_lid_parallel -nx 1000 -ny 1000 -o 1.2 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple velocity
# echo "Completed\n"

echo "Initiating flow pattern comparison by varying grid size to simulate different Reynold's number" 
mpiexec -n 100 python3 -m simulations.sliding_lid_parallel -u 0.05 -o 0.8 -re_min 100 -re_max 1000 -re_count 10 -nt 5 -run_multiple grid
echo "Completed\n"