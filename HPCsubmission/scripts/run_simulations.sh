echo "Running all simulations in order"
echo "--------------------------------"

echo "Shear wave decay simulation..."
echo "--------------------------------"
echo "Initiating run with sinusoidal density"
python3 -m simulations.shear_wave_decay -o 1.2 -rho_o 1 -rho_epsilon 0.01 -nx 50 -ny 50 -nt 5600 -nt_log 400
echo "Completed"

echo "Initiating run with sinusoidal velocity"
python3 -m simulations.shear_wave_decay -o 1.2 -u_epsilon 0.01 -nx 50 -ny 50 -nt 5600 -nt_log 400
echo "Completed"

echo "Initiating omega comparison for sinusoidal velocity"
python3 -m simulations.shear_wave_decay -u_epsilon 0.01 -nx 50 -ny 50 -nt 4000 -omega_comparison
echo "Completed"

echo "Initiating omega comparison for sinusoidal density"
python3 -m simulations.shear_wave_decay -rho_o 1 -rho_epsilon 0.01 -nx 50 -ny 50 -nt 4000 -omega_comparison
echo "Completed"

echo "Couette flow simulation..."
echo "--------------------------------"
python3 -m simulations.couette_flow -o 1.2 -u 0.01 -nx 100 -ny 100 -nt 50000 -nt_log 5000
echo "Completed"

echo "Poiseuille flow simulation..."
echo "--------------------------------"
python3 -m simulations.poiseuille_flow -o 1.0 -rho_in 1.001 -rho_out 0.999 -nx 100 -ny 100 -nt 200000
echo "Completed"

echo "Sliding Lid simulation (in series)..."
echo "--------------------------------"
python3 -m simulations.sliding_lid -o 1.2 -nx 300 -ny 300 -nt 50000 -nt_log 5000
echo "Completed"

echo "Sliding Lid simulation (in parallel)..."
echo "--------------------------------"
mpiexec -n 4 python3 -m simulations.sliding_lid_parallel -o 1.2 -nx 300 -ny 300 -nt 50000 -nt_log 5000
echo "Completed"
