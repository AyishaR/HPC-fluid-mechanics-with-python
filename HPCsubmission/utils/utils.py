import os
import pandas as pd
from utils.constants import *

def write_time_to_file(path, cores, grid_size, time_step, time_value):
    """
    Write/Append the given values to the csv file in 'path'. These will be used to plot the 'MLUPS vs processor count' comparison plot later.

    :param path: Path to csv file
    :type path: str
    :param cores: Number of processors used to compute
    :type cores: int
    :param grid_size: Lattice grid size (Assume square grid size)
    :type grid_size: int
    :param time_step: Number of time steps simulated
    :type time_step: int
    :param time_value: The time of execution (in seconds)
    :type time_value: float
    """
    columns = ["Cores", "Grid", "Time", "Timestep"]
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
    if len(df[(df['Grid'] == grid_size)&(df['Cores'] == cores)&(df['Timestep']==time_step)])>0:
        df.loc[(df['Grid'] == grid_size)&(df['Cores'] == cores)&(df['Timestep']==time_step), "Time"] = time_value
    else:
        new_row = pd.DataFrame({'Cores': cores, 
                                'Grid': [grid_size], 
                                "Time": [time_value],
                                "Timestep": time_step})
        df = pd.concat([df, new_row])

    df = df[columns]
    df.to_csv(path)

        