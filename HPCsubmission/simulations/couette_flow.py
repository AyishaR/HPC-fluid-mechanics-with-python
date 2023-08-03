import argparse
import math
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *
from utils.boundaries import *

class CouetteFlow:
    def __init__(self) -> None:
        args = self.parse()

        self.nx = args.nx
        self.ny = args.ny
        self.L = self.ny
        self.X, self.Y = np.meshgrid(np.arange(0,self.nx), np.arange(0,self.ny))
        
        self.omega = args.o

        self.ub = args.u
        self.vb = 0
        self.velocity_boundary = np.array([self.ub,self.vb])

        self.subplot_columns = args.plot_grid
        self.nt = args.nt
        self.nt_log = args.nt_log

        self.config_title = ""
        self.path = f"plots/CouetteFlow/{self.config_title}"

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-nx', type=int, default=100,
                            help='Grid size along X-axis')
        parser.add_argument('-ny', type=int, default=100,
                            help='Grid size along Y-axis')
        parser.add_argument('-o', type=float, default=1.2,
                            help='Omega')
        parser.add_argument('-u', type=float, default=0.01,
                            help='Horizontal velocity of the top wall')
        parser.add_argument('-plot_grid', type=int, default=5,
                            help='Number of density plots in one row')
        parser.add_argument('-nt', type=int, default=10000,
                            help='Number of timesteps')
        parser.add_argument('-nt_log', type=int, default=1000,
                            help='Timestep interval to record/plot values')
        args = parser.parse_args()
        return args

    def couette_flow(self, f_inm):

        # Streaming
        f_inm = stream(f_inm)
        
        # Wall
        f_inm = top_boundary(f_inm, moving=True, ub=self.ub, vb=self.vb)
        f_inm = bottom_boundary(f_inm)
        
        # Collision
        f_inm = collision(f_inm, self.omega)

        return f_inm
    
    def simulate_couette_flow(self,
                              f,
                              plot=True):
        slen = math.ceil(self.nt/self.nt_log)
        u_periodic = np.empty((slen, self.ny))
        r_periodic = np.empty((slen, self.ny))
        u_amplitude = np.empty((slen))
        r_amplitude = np.empty((slen))
        
        if plot:
            subplot_rows = math.ceil(slen/self.subplot_columns)
            fig, axes = plt.subplots(subplot_rows, self.subplot_columns)
            plt.setp(axes, xticks=range(0,self.nx+1,self.nx//5), yticks=range(0,self.ny+1,self.ny//5))
            plt.gcf().set_size_inches(self.subplot_columns*3,subplot_rows*3)
            if self.config_title:
                plt.suptitle(self.config_title)

        for i in range(self.nt):
            f = self.couette_flow(f)
            if i%self.nt_log==0:
                print(i)
                rho = get_rho(f[:,1:-1,1:-1])
                u = get_u(f[:,1:-1,1:-1],rho)
                idx = math.ceil(i/self.nt_log)
                u_periodic[idx] = u[0,self.nx//2,:]
                u_amplitude[idx] = max(u_periodic[idx])
                r_periodic[idx] = rho[self.nx//2,:]
                r_amplitude[idx] = max(r_periodic[idx])
                
                if plot:
                    axis = axes[math.floor(idx/self.subplot_columns), idx%self.subplot_columns]

                    line_top = lines.Line2D([-0.5, -0.5+self.nx],
                            [-0.5, -0.5],
                            color ='blue')
                    line_bottom = lines.Line2D([-0.5, -0.5+self.nx],
                            [-0.5+self.ny, -0.5+self.ny],
                            color ='blue')
        
                    plot_density(rho, axis, f"Step {i}", [line_top, line_bottom])
                    axis.streamplot(self.X, self.Y, u[0].T, u[1].T,color='white')

        if plot:
            plt.savefig(f"{self.path}/{self.config_title}/Streamplot.png")
        return u_periodic, r_periodic, u_amplitude, r_amplitude
    
    def amplitude_combined_plot(self, u_periodic):
        # Combined plot
        for i in range(u_periodic.shape[0]):
            plt.plot(u_periodic[i][1:-1])
        plt.title(f"Config: omega={self.omega}; y=0:Rigid wall; y=50:Moving wall with u={self.ub};")
        plt.xticks(range(0,self.nx+1, self.nx//5))
        plt.xlabel("Y-coordinate")
        plt.ylabel("Velocity")
        plt.show()
        plt.savefig(f"{self.path}/{self.config_title}/Amplitude_combined_plot.png")

    def run(self):
        self.config_title = f"Omega-{self.omega};ub-{self.ub};vb-{self.vb}"
        os.makedirs(f"{self.path}/{self.config_title}", exist_ok=True)
        
        f = np.einsum("i,jk->ijk", W_I, np.ones((self.nx+2, self.ny+2)))
        u_periodic, r_periodic, u_amplitude, r_amplitude = \
        self.simulate_couette_flow(f)

        # Wave decay
        plot_decay(u_periodic[::2,1:-1], 
                   "Y-coordinate", "Velocity",
                   f"Velocity variation plot - y=0:Rigid wall; y={self.ny}:Moving wall with u={self.ub}; "+self.config_title, 
                   path=f"{self.path}/{self.config_title}")
        
        # Combined amplitude plot
        plot_combined(u_periodic,
                      "Y-coordinate", "Velocity",
                      f"{self.path}/{self.config_title}/Amplitude_combined_plot.png",
                      title=self.config_title)


swd = CouetteFlow()
swd.run()