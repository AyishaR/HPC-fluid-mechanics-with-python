#!/bin/bash
N=4
O="1.2"
U="0.01"
NT="5000"

TIMELOG_PATH="reports/serial_vs_parallel.csv"

grid_sizes=(
    50 100 150 200 250 300
)

for values in "${grid_sizes[@]}"; do
    echo "---------------Velocity: $values-----------------"
    echo "Serial implementation intitiating"
    python3 -m simulations.sliding_lid -nx "$values" -ny "$values" -o "$O" -u "$U" -nt "$NT" -nt_log "$NT" -no_plot -time_log_path "$TIMELOG_PATH"
    echo "Serial implementation completed"

    echo "Parallel implementation intitiating"
    mpiexec -n $N python3 -m simulations.sliding_lid_parallel -nx "$values" -ny "$values" -o "$O" -u "$U" -nt "$NT" -nt_log "$NT" -no_plot -time_log_path "$TIMELOG_PATH"
    echo "Parallel implementation completed"
done

echo "Intitiating comparison plot"
python3 -m scaling_test.serial_vs_parallel -time_log_path "$TIMELOG_PATH" -config_title "Omega-$O-u-$U-nt-$NT" 
echo "Completed comparison plot"