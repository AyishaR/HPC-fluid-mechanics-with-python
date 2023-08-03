import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.parallelization import *
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *
from utils.boundaries import *

class SlidingLid(Parallelization):
    def __init__(self) -> None:
        super().__init__()
        args = self.parse()

        self.nx = args.nx
        self.ny = args.ny
        self.L = self.ny
        self.X, self.Y = np.meshgrid(np.arange(0,self.nx), np.arange(0,self.ny))
        
        self.omega = args.o

        self.ub = args.u
        self.vb = 0
        self.velocity_boundary = np.array([self.ub,self.vb])

        self.min_u = 0.01
        self.max_u = 0.1
        self.u_count = 5

        self.subplot_columns = args.plot_grid
        self.nt = args.nt
        self.nt_log = args.nt_log

        self.grid_division()

        self.config_title = ""
        self.path = f"plots/SlidingLidParallel/{self.config_title}"

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
    
    def sliding_lid(self, f_inm, boundaries=None):
        
        # Streaming
        f_inm = stream(f_inm)
        
        # Wall
        if boundaries:
            for b in boundaries:
                if b==LEFT:
                    f_inm = left_boundary(f_inm)
                    continue
                if b==RIGHT:
                    f_inm = right_boundary(f_inm)
                    continue
                if b==BOTTOM:
                    f_inm = bottom_boundary(f_inm)
                    continue
                if b==TOP:
                    f_inm = top_boundary(f_inm, moving=True, ub=self.ub, vb=self.vb)
            
        # Collision 
        f_inm = collision(f_inm, self.omega)

        return f_inm
    
    def simulate_sliding_lid(self,
                              f_all,
                              plot=True):
        slen = math.ceil(self.nt/self.nt_log)
        f = np.zeros((9,self.nxsub,self.nysub))
        xlo, xhi, ylo, yhi = self.get_coordinates_of_rank()
        f[:,1:-1,1:-1] = f_all[:,xlo:xhi,ylo:yhi]
        boundaries = self.get_boundaries()
        
        if plot and self.rank==0:
            subplot_rows = math.ceil(slen/self.subplot_columns)
            fig, axes = plt.subplots(subplot_rows, self.subplot_columns)
            plt.setp(axes, xticks=range(0,self.nx+1,self.nx//5), yticks=range(0,self.ny+1,self.ny//5))
            plt.gcf().set_size_inches(self.subplot_columns*3,subplot_rows*3+1)
            if self.config_title:
                plt.suptitle(self.config_title)

        for i in range(self.nt):
            f = self.communicate(f)
            f = self.sliding_lid(f, boundaries)
            if i%self.nt_log==0:
                print(i)
                c_full_range = np.zeros((Q*self.nx*self.ny))
                self.comm.Gather(f[:,1:-1,1:-1].reshape(9*(self.nxsub-2)*(self.nysub-2)), c_full_range, root = 0)
                
                if self.rank == 0:
                    print(i)
                    c_plot = np.zeros((Q,self.nx,self.ny))
                    
                    for r in range(self.size):
                        xlo, xhi, ylo, yhi = self.get_coordinates_of_rank(r)
                        clo = r*self.nx*self.ny//(self.sectsX*self.sectsY)
                        chi = (r+1)*self.nx*self.ny//(self.sectsX*self.sectsY)
                        c_plot[:,xlo:xhi,ylo:yhi] = c_full_range[Q*clo:Q*chi].reshape(Q,self.nx//self.sectsX,self.ny//self.sectsY)
            
                    rho = get_rho(c_plot)
                    u = get_u(c_plot,rho)
                    idx = math.ceil(i/self.nt_log)

                    if plot:
                        axis = axes[math.floor(idx/self.subplot_columns), idx%self.subplot_columns]

                        plot_density(rho, axis, f"Step {i}")
                        axis.streamplot(self.X, self.Y, u[0].T, u[1].T,color='white')
        if plot:
            plt.show()
        
        c_full_range = np.zeros((Q*self.nx*self.ny))
        self.comm.Gather(f[:,1:-1,1:-1].reshape(Q*(self.nxsub-2)*(self.nysub-2)), c_full_range, root = 0)
        if self.rank == 0:
            c_plot = np.zeros((Q,self.nx,self.ny))
            for r in range(self.size):
                xlo, xhi, ylo, yhi = self.get_coordinates_of_rank(r)
                clo = r*self.nx*self.ny//(self.sectsX*self.sectsY)
                chi = (r+1)*self.nx*self.ny//(self.sectsX*self.sectsY)
                c_plot[:,xlo:xhi,ylo:yhi] = c_full_range[Q*clo:Q*chi].reshape(Q,self.nx//self.sectsX,self.ny//self.sectsY)
                
            return c_plot
            
    
    def get_reynolds_number(self, L, u, o):
        nu = (1/3)*((1/o)-0.5)
        return (L*u)/nu
    
    def run_multiple_velocity(self):
        self.config_title = f"Omega-{self.omega};"
        os.makedirs(f"{self.path}/{self.config_title}", exist_ok=True)

        velocity_list = np.round(np.linspace(self.min_u, self.max_u, self.u_count),2)
        # reynolds_list = np.zeros((len(velocity_list), 2))

        if self.rank==0:
            slen = len(velocity_list)
            subplot_rows = math.ceil(slen/self.subplot_columns)
            fig, axes = plt.subplots(subplot_rows, self.subplot_columns)
            plt.setp(axes, xticks=range(0,self.nx+1,self.nx//5), yticks=range(0,self.ny+1,self.ny//5))
            plt.gcf().set_size_inches(self.subplot_columns*3,subplot_rows*3)

        for idx, value in enumerate(velocity_list):
            rn = np.round(self.get_reynolds_number(self.L, value, self.omega))
            config_title = f"Reynolds number: {rn}"
            
            f = np.einsum("i,jk->ijk", W_I, np.ones((self.nx,self.ny)))
            self.ub = value
            f = self.simulate_sliding_lid(f, plot=False)
            if self.rank==0:
                rho = get_rho(f)
                u = get_u(f,rho)

                axis = axes[math.floor(idx/self.subplot_columns), idx%self.subplot_columns]

                plot_density(rho, axis, config_title)
                axis.streamplot(self.X, self.Y, u[0].T, u[1].T, color='white')
        plt.suptitle(self.config_title)
        plt.show()
        plt.savefig(f"{self.path}/Reynolds_number_comparison_{self.min_u}_{self.max_u}_{self.u_count}.png")
    
    
    def run(self):
        self.config_title = f"Omega-{self.omega};u-{self.ub};"
        os.makedirs(f"{self.path}/{self.config_title}", exist_ok=True)
        
        f = np.einsum("i,jk->ijk", W_I, np.ones((self.nx+2, self.ny+2)))
        _ = self.simulate_sliding_lid(f)

        # # Wave decay
        # self.combined_plot(r_periodic,  "Y-coordinate", "Density",
        #                    f"{self.path}/{self.config_title}",
        #                    None, f"Density variation plot -  "+self.config_title)
        
        # analytical_u = self.get_analytical_velocity()
        # # Combined velocity plot
        # self.combined_plot(u_periodic,  "Y-coordinate", "Velocity",
        #                    f"{self.path}/{self.config_title}",
        #                    analytical_u, f"Velocity variation plot -  "+self.config_title)
        
        # # Combined velocity plot
        # self.combined_plot(u_inlet, "Y-coordinate", "Velocity (inlet)",
        #                    f"{self.path}/{self.config_title}",
        #                    analytical_u, f"Velocity variation plot (inlet) -  "+self.config_title)


swd = SlidingLid()
swd.run_multiple_velocity()