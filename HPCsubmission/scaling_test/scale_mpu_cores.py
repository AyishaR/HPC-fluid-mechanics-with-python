"""Simulate the Sliding Lid for the same set of parameters in serial and parallel and compare runtimes.
As the runtime is directly related to the dimensions of the lattice grid, the nx and ny values will be varied across different runs.
"""
import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpi4py import MPI
from simulations.sliding_lid import SlidingLid as SL_serial
from simulations.sliding_lid_parallel import SlidingLid as SL_parallel
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *
from utils.boundaries import *

class ScaleMPUs:
    def __init__(self) -> None:
        args = self.parse()

        self.time_log_path = args.time_log_path
        self.plot_path = args.plot_path
        self.config_title = args.config_title

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-time_log_path', type=str, 
                            default="reports/report.csv",
                            help='Path to report with comparison of time of execution')
        parser.add_argument('-plot_path', type=str, 
                            default="plots/Scale_mpus",
                            help='Path to save plot with comparison of time of execution')
        parser.add_argument('-config_title', type=str, 
                            default="",
                            help='Config of the run to be included in the title of the plot')
        args = parser.parse_args()
        return args
    
    def run(self):
        print("Running", self.time_log_path, ";")
        if os.path.exists(self.time_log_path):
            os.makedirs(f"{self.plot_path}", exist_ok=True)
            df = pd.read_csv(self.time_log_path)
            df = df.sort_values(by=['Grid','Cores'], ascending=True)
            df['MLUPS'] = df.apply(lambda x: round((x['Timestep']*x['Grid']*x['Grid'])/(x[PARALLEL]*1000000),2), axis=1)
            # df['proc_log10'] = df.apply(lambda x: round(np.)
            unique_grids = df['Grid'].unique()
            for grid_value in unique_grids:
                plt.loglog(df[df['Grid']==grid_value]['Cores'].tolist(), 
                         df[df['Grid']==grid_value]['MLUPS'].tolist(), 
                         marker='o', linestyle='-', 
                         label=f'Grid {grid_value}x{grid_value}')
            # plt.axis("equal")
            plt.xlabel('Number of cores')
            plt.ylabel('Parallel Execution Time (seconds)')
            plt.title(f'Comparison of execution time with respect to number of cores - {self.config_title}', size=10, wrap=True)
            plt.legend()

            plt.show(block=True)
            plt.savefig(f"{self.plot_path}/Plot.png")
            plt.clf()
            plt.cla()
            plt.close()

if __name__ == "__main__":
    sp = ScaleMPUs()
    sp.run()