"""Plot results of scale test from CSV file.
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *
from utils.boundaries import *

class ScaleMPUsVisualize:
    """
    Class to plot results of the scale test from a CSV file that has run time, processor counts and other parameters.
    """
    def __init__(self) -> None:
        """
        Initialize instance variables.
        """
        args = self.parse()

        self.time_log_path = args.time_log_path
        self.plot_path = args.plot_path
        self.config_title = args.config_title

    def parse(self):
        """
        Argument parser to parse command line arguments and assign defaults.

        :return: Arguments of the class
        :rtype: dict
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-time_log_path', type=str, 
                            default="reports/report.csv",
                            help='Path to report with comparison of time of execution')
        parser.add_argument('-plot_path', type=str, 
                            default="plots/Scale_cores_mpus",
                            help='Path to save plot with comparison of time of execution')
        parser.add_argument('-config_title', type=str, 
                            default="",
                            help='Config of the run to be included in the title of the plot')
        args = parser.parse_args()
        return args
    
    def run(self):
        """
        Plot the results of scale test. Calculate MLUPS and plot against number of cores for different grid sizes. The plot is a log plot.
        """
        if os.path.exists(self.time_log_path):
            os.makedirs(f"{self.plot_path}", exist_ok=True)
            df = pd.read_csv(self.time_log_path)
            df = df.sort_values(by=['Grid','Cores'], ascending=True)
            df['MLUPS'] = df.apply(lambda x: round((x['Timestep']*x['Grid']*x['Grid'])/(x["Time"]*1000000),2), axis=1)
            # df['proc_log10'] = df.apply(lambda x: round(np.)
            unique_grids = df['Grid'].unique()
            for grid_value in unique_grids:
                plt.loglog(df[df['Grid']==grid_value]['Cores'].tolist(), 
                         df[df['Grid']==grid_value]['MLUPS'].tolist(), 
                         marker='o', linestyle='-', 
                         label=f'Grid {grid_value}x{grid_value}')
            plt.xlabel('Number of processors')
            plt.ylabel('MLUPS')
            plt.title(f'MLUPS vs number of processors - {self.config_title}', size=10, wrap=True)
            plt.legend()
            plt.show(block=False)
            plt.savefig(f"{self.plot_path}/Plot_MLUPS.png")
            plt.clf()
            plt.cla()
            plt.close()

if __name__ == "__main__":
    scale_visualize = ScaleMPUsVisualize()
    scale_visualize.run()   