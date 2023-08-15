import os
import pandas as pd
from utils.constants import *

def write_time_to_file(path, cores, grid_size, time_step, time_value, mode):
    columns = ["Cores", "Grid", SERIAL, PARALLEL, "Timestep"]
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
    if len(df[(df['Grid'] == grid_size)&(df['Cores'] == cores)&(df['Timestep']==time_step)])>0:
        df.loc[(df['Grid'] == grid_size)&(df['Cores'] == cores)&(df['Timestep']==time_step), mode] = time_value
    else:
        new_row = pd.DataFrame({'Cores': cores, 
                                'Grid': [grid_size], 
                                mode: [time_value],
                                "Timestep": time_step})
        df = pd.concat([df, new_row])

    df = df[columns]
    df.to_csv(path)

        