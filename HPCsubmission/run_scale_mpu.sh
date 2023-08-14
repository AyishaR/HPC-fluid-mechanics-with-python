#!/bin/bash
O="1.2"
U="0.01"
NT="5000"
NX="300"
NY="300"

TIMELOG_PATH="reports/scale_mpu_cores.csv"

mpu_cores=(
    1 2 3 4 4 5 6 7 8 9 10
)

for values in "${mpu_cores[@]}"; do
    echo "---------------Omega: $values-----------------"
    mpiexec -n $values python3 -m simulations.sliding_lid_parallel -nx "$NX" -ny "$NY" -o "$O" -u "$U" -nt "$NT" -nt_log "$NT" -no_plot -time_log_path "$TIMELOG_PATH"
done

echo "Intitiating comparison plot"
python3 -m scaling_test.scale_mpu_cores -time_log_path "$TIMELOG_PATH" -config_title "Omega-$O-u-$U-nt-$NT-GridSize-$NX x $NY" 
echo "Completed comparison plot"