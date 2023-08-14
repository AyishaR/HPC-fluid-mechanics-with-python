#!/bin/bash
O="1.2"
U="0.01"
NT="50000"
NX="1000"
NY="1000"

TIMELOG_PATH="reports/scale_mpu_cores.csv"

mpu_cores=(
    2 4 8 16 32 64
)

for values in "${mpu_cores[@]}"; do
    echo "---------------Core: $values-----------------"
    mpiexec -n $values python3 -m simulations.sliding_lid_parallel -nx "$NX" -ny "$NY" -o "$O" -u "$U" -nt "$NT" -nt_log "$NT" -no_plot -time_log_path "$TIMELOG_PATH"
done

# echo "Intitiating comparison plot"
# python3 -m scaling_test.scale_mpu_cores -time_log_path "$TIMELOG_PATH" -config_title "Omega-$O-u-$U-nt-$NT-GridSize-$NX x $NY" 
# echo "Completed comparison plot"