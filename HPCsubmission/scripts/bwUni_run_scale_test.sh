# #!/bin/bash
# #SBATCH --nodes=3
# #SBATCH --ntasks-per-node=40
# #SBATCH --mem=90000
# #SBATCH --time=00:30:00
# #SBATCH --partition=dev_multiple
# #SBATCH --output=run_scale_test.out
# #SBATCH --error=run_scale_test.err
# #SBATCH --export=ALL

module load compiler/gnu/10.2   
module load mpi/openmpi/4.1
module load devel/python/3.8.6_gnu_10.2

O="1.2"
U="0.01"
NT="50000"

TIMELOG_PATH="reports/scale_test.csv"

grid_size=(
    100 500 1000
)
cores=(
    16 25 100 400 625 2500
)

for grid_val in "${grid_size[@]}"; do
    for proc_val in "${cores[@]}"; do
        echo "---------------Grid: $grid_val | Core: $proc_val-----------------"
        mpiexec -n $proc_val python3 -m simulations.sliding_lid_parallel -nx "$grid_val" -ny "$grid_val" -o "$O" -u "$U" -nt "$NT" -nt_log "$NT" -no_plot -time_log_path "$TIMELOG_PATH"
    done
done

# echo "Intitiating comparison plot"
# python3 -m scaling_test.scale_test_visualize -time_log_path "$TIMELOG_PATH" -config_title "Omega-$O  u-$U  nt-$NT" 
# echo "Completed comparison plot"