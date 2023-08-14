import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import timeit
from tqdm import tqdm
from utils.parallelization import *
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *
from utils.boundaries import *
from utils.utils import *

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

        self.re_min = args.re_min
        self.re_max = args.re_max
        self.re_count = args.re_count

        self.run_multiple_omega_flag = args.run_multiple_omega
        self.run_multiple_velocity_flag = args.run_multiple_velocity

        self.subplot_columns = args.plot_grid
        self.nt = args.nt
        self.nt_log = args.nt_log

        if args.no_plot:
            self.plot = False
        else:
            self.plot = True

        self.time_log_path = args.time_log_path
        self.varying_parameter = args.run_multiple

        if self.varying_parameter!="grid":
            self.grid_division()

        self.config_title = ""
        self.path = f"plots/SlidingLidParallel/{self.config_title}"

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-nx', type=int, default=300,
                            help='Grid size along X-axis')
        parser.add_argument('-ny', type=int, default=300,
                            help='Grid size along Y-axis')
        parser.add_argument('-o', type=float, default=1.2,
                            help='Omega')
        parser.add_argument('-u', type=float, default=0.01,
                            help='Horizontal velocity of the top wall')
        parser.add_argument('-re_min', type=int, default=100,
                            help='Minimum Reynolds number for comparison run')
        parser.add_argument('-re_max', type=int, default=1000,
                            help='Maximum Reynolds number for comparison ru')
        parser.add_argument('-re_count', type=int, default=10,
                            help='Number of Reynolds number values to pick between minimum and maximum range')
        parser.add_argument('-plot_grid', type=int, default=5,
                            help='Number of density plots in one row')
        parser.add_argument('-nt', type=int, default=100000,
                            help='Number of timesteps')
        parser.add_argument('-nt_log', type=int, default=10000,
                            help='Timestep interval to record/plot values')
        parser.add_argument("-no_plot", action="store_true", help="Enable plotting")
        parser.add_argument("-time_log_path", type=str, default="",
                            help='Path to file to log execution time')
        parser.add_argument("-run_multiple", type=str, default="",
                            help='Parameter to vary while running comparison')
        parser.add_argument("-run_multiple_velocity", action="store_true", help="Enable Reynolds number comparison for multiple velocity initiation")
        parser.add_argument("-run_multiple_omega", action="store_true", help="Enable Reynolds number comparison for multiple omega initiation")
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
                              plot=None):
        
        if plot is None:
            plot = self.plot
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
                plt.suptitle(f"Sliding Lid - {self.config_title}", wrap=True)

        for i in tqdm(range(self.nt)):
            f = self.communicate(f)
            f = self.sliding_lid(f, boundaries)
            if i%self.nt_log==0:
                c_full_range = np.zeros((Q*self.nx*self.ny))
                self.comm.Gather(f[:,1:-1,1:-1].reshape(9*(self.nxsub-2)*(self.nysub-2)), c_full_range, root = 0)
                
                if self.rank == 0:
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

                        rectangle = Rectangle((-0.5,-0.5), 
                                                self.nx-0.5, 
                                                self.ny-0.5, 
                                                fill = False, color='black',
                                                linewidth=5)
                        line_top = lines.Line2D([-0.5, -0.5+self.nx],
                                [-0.5+self.ny, -0.5+self.ny],
                                color ='red',
                                linewidth=5) 

                        plot_density(rho, axis, f"Time step {i}", [rectangle, line_top])
                        axis.streamplot(self.X, self.Y, u[0].T, u[1].T,color='white')
        
        if plot and self.rank==0:
            plt.show(block=False)
            plt.savefig(f"{self.path}/{self.config_title}/Streamplot.png")
            plt.clf()
            plt.cla()
            plt.close()
        
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
            
    def get_nu_from_omega(self, o):
        return (1/3)*((1/o)-0.5)
    
    def get_omega_from_nu(self, nu):
        return 1/(3*nu +0.5)
    
    def get_reynolds_number(self, L, u, o):
        return (L*u)/self.get_nu_from_omega(o)
    
    def get_L(self, r, u, o):
        return (r*self.get_nu_from_omega(o))/u
    
    def get_o(self, r, L, u):
        nu = (L*u)/r
        return self.get_omega_from_nu(nu)
    
    def get_u(self, r, L, o):
        return (r*self.get_nu_from_omega(o))/L
    
    def run_multiple_reynolds_number(self):
        assert (self.varying_parameter in [GRID, VELOCITY, OMEGA])
        os.makedirs(f"{self.path}/Varying_{self.varying_parameter}", exist_ok=True)

        re_list = np.round(np.linspace(self.re_min, self.re_max, self.re_count),2)

        if self.rank==0:
            slen = len(re_list)
            subplot_rows = math.ceil(slen/self.subplot_columns)
            fig, axes = plt.subplots(subplot_rows, self.subplot_columns)
            plt.setp(axes, xticks=range(0,self.nx+1,self.nx//5), yticks=range(0,self.ny+1,self.ny//5))
            plt.gcf().set_size_inches(self.subplot_columns*3,subplot_rows*3)

        for idx, value in enumerate(re_list):
            if self.varying_parameter == GRID:
                L = int(self.get_L(value, self.ub, self.omega))
                self.nx = self.ny = self.L = L
                self.X, self.Y = np.meshgrid(np.arange(0,self.nx), np.arange(0,self.ny))
                self.grid_division()

                subplot_title = f"Re: {value} (L- {self.L})"
                self.config_title = f"Reynolds number comparison with varying grid size (\u03A9- {self.omega}; u- {self.ub})"

            elif self.varying_parameter == OMEGA:
                self.omega = np.round(self.get_o(value, self.L, self.ub),2)

                subplot_title = f"Re- {value} (\u03A9: {self.omega})"
                self.config_title = f"Reynolds number comparison with varying grid size (L- {self.L}; u- {self.ub})"

            elif self.varying_parameter == VELOCITY:
                self.ub = np.round(self.get_u(value, self.L, self.omega),2)

                subplot_title = f"Re- {value} (u- {self.ub})"
                self.config_title = f"Reynolds number comparison with varying grid size (L- {self.L}; \u03A9- {self.omega})"
   
            f = np.einsum("i,jk->ijk", W_I, np.ones((self.nx,self.ny)))

            f = self.simulate_sliding_lid(f, plot=False)
            
            if self.rank==0:
                rho = get_rho(f)
                u = get_u(f,rho)

                axis = axes[math.floor(idx/self.subplot_columns), idx%self.subplot_columns]

                rectangle = Rectangle((-0.5,-0.5), 
                                        self.nx-0.5, 
                                        self.ny-0.5, 
                                        fill = False, color='black',
                                        linewidth=5)
                line_top = lines.Line2D([-0.5, -0.5+self.nx],
                        [-0.5+self.ny, -0.5+self.ny],
                        color ='red',
                        linewidth=5) 

                plot_density(rho, axis, subplot_title, [rectangle, line_top])
                plt.setp(axis, xticks=range(0,self.nx+1,self.nx//5), yticks=range(0,self.ny+1,self.ny//5))
                axis.streamplot(self.X, self.Y, u[0].T, u[1].T, color='white')
        plt.suptitle(self.config_title)
        plt.show(block=False)
        plt.savefig(f"{self.path}Varying_{self.varying_parameter}/{self.config_title}.png")
        plt.clf()
        plt.cla()
        plt.close()

    def run(self):
        self.config_title = f"Omega-{self.omega};u-{self.ub};"
        os.makedirs(f"{self.path}/{self.config_title}", exist_ok=True)
        
        f = np.einsum("i,jk->ijk", W_I, np.ones((self.nx+2, self.ny+2)))

        start_time = time.time()
        _ = self.simulate_sliding_lid(f)
        end_time = time.time()

        if self.rank==0 and self.time_log_path:
            write_time_to_file(self.time_log_path, self.size, self.nx, round(end_time-start_time, 2), PARALLEL)


if __name__ == "__main__":
    s_lid = SlidingLid()
    if s_lid.run_multiple_velocity_flag:
        execution_time = timeit.timeit(s_lid.run_multiple_velocity, number=1)
        print("Execution time:", execution_time, "seconds")
    elif s_lid.run_multiple_omega_flag:
        execution_time = timeit.timeit(s_lid.run_multiple_omega, number=1)
        print("Execution time:", execution_time, "seconds")
    elif s_lid.varying_parameter:
        execution_time = timeit.timeit(s_lid.run_multiple_reynolds_number, number=1)
        print("Execution time:", execution_time, "seconds")
    else:
        execution_time = timeit.timeit(s_lid.run, number=1)
        print("Execution time:", execution_time, "seconds")