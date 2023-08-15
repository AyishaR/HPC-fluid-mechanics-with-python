import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *

class ShearWaveDecay:
    def __init__(self) -> None:
        """
        Initialize the instance variables.
        """
        args = self.parse()

        self.nx = args.nx
        self.ny = args.ny
        self.L = self.ny
        self.X, self.Y = np.meshgrid(np.arange(0,self.nx), np.arange(0,self.ny))

        self.omega = args.o

        self.rho_o = args.rho_o
        self.rho_epsilon = args.rho_epsilon
        self.u_epsilon = args.u_epsilon

        self.omega_min = args.omega_min
        self.omega_max = args.omega_max
        self.omega_count = args.omega_count

        self.omega_comparison = args.omega_comparison
        
        self.subplot_columns = args.plot_grid
        self.nt = args.nt
        self.nt_log = args.nt_log

        self.config_title = ""
        self.path = f"plots/ShearWaveDecay"

    def parse(self):
        """
        Argument parser to parse command line arguments and assign defaults.

        :return: Arguments of the class
        :rtype: dict
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-nx', type=int, default=300,
                            help='Grid size along X-axis')
        parser.add_argument('-ny', type=int, default=300,
                            help='Grid size along Y-axis')
        parser.add_argument('-o', type=float, default=1.2,
                            help='Omega')
        parser.add_argument('-u_epsilon', type=float, default=None,
                            help='Horizontal velocity factor')
        parser.add_argument('-rho_o', type=float, default=None,
                            help='Initial density')
        parser.add_argument('-rho_epsilon', type=float, default=None,
                            help='Horizontal density factor')
        parser.add_argument('-omega_min', type=float, default=0.1,
                            help='Minimum omega for comparison run')
        parser.add_argument('-omega_max', type=float, default=1.5,
                            help='Maximum omega for comparison ru')
        parser.add_argument('-omega_count', type=int, default=15,
                            help='Number of omega values to pick between minimum and maximum range')
        parser.add_argument('-plot_grid', type=int, default=5,
                            help='Number of density plots in one row')
        parser.add_argument('-nt', type=int, default=10000,
                            help='Number of timesteps')
        parser.add_argument('-nt_log', type=int, default=1000,
                            help='Timestep interval to record/plot values')
        parser.add_argument("-omega_comparison", action="store_true", help="Enable omega comparison for sinusoidal velocity initiation")
        args = parser.parse_args()
        return args
    
    def shear_wave_decay(self, f_inm):
        """
        Simulate one time step of Shear Wave Decay.

        :param f_inm: Particle probability density 
        :type f_inm: np.array
        :return: Particle probability density after one time step simulation
        :rtype: np.array
        """
        # Streaming
        f_inm = stream(f_inm)
        
        # Collision
        f_inm = collision(f_inm, self.omega)
    
        return f_inm
    
    def simulate_shear_wave_decay(self,
                                  f,
                                  plot=True):
        """
        Simulate shear wave decay and plot density if applicable.

        :param f: Particle probability density
        :type f: np.array
        :param plot: Flag on whether to plot, defaults to True
        :type plot: bool, optional
        :return: Values logged at periodic timesteps - u_periodic, r_periodic, u_amplitude, r_amplitude
        :rtype: np.array, np.array, np.array, np.array
        """
        slen = math.ceil(self.nt/self.nt_log)+1
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
                plt.suptitle(f"Shear Wave Decay - {self.config_title}", wrap=True)

        for i in tqdm(range(self.nt+1)):
            f = self.shear_wave_decay(f)
            if i%self.nt_log==0:
                rho = get_rho(f)
                u = get_u(f,rho)
                idx = math.ceil(i/self.nt_log)
                u_periodic[idx] = u[0,self.nx//2,:]
                u_amplitude[idx] = max(u_periodic[idx])
                r_periodic[idx] = rho[self.nx//2,:]
                r_amplitude[idx] = max(r_periodic[idx])
                
                if plot:
                    axis = axes[math.floor(idx/self.subplot_columns), idx%self.subplot_columns]
        
                    plot_density(rho, axis, f"Step {i}")
                    axis.streamplot(self.X, self.Y, u[0].T, u[1].T,color='white')

        if plot:
            plt.show(block=False)
            plt.savefig(f"{self.path}/{self.config_title}/Streamplot.png")
            plt.clf()
            plt.cla()
            plt.close()
        return u_periodic, r_periodic, u_amplitude, r_amplitude
    
    def run_multiple_omega(self):
        """
        Run Shear Wave Decay for multiple omega values and plot comparison plots.
        """
        omega_list = np.round(np.linspace(self.omega_min,
                                          self.omega_max,
                                          self.omega_count),2)
        viscosities = np.zeros((len(omega_list), 3))

        if self.rho_o is not None and self.rho_epsilon is not None:
            rho_init = self.rho_o + self.rho_epsilon*np.sin((2.*np.pi*self.X)/self.L)
            self.config_title += \
            f" Sinusoidal density - rho_o-{self.rho_o};rho_epsilon-{self.rho_epsilon};"
        else:
            rho_init = np.ones((self.nx, self.ny))

        if self.u_epsilon is not None:
            ux = self.u_epsilon*np.sin((2.*np.pi*self.X)/self.L)
            uy = np.zeros((self.nx, self.ny))
            u_init = np.stack((ux,uy), axis = 0)
            self.config_title += f" Sinusoidal velocity - u_epsilon-{self.u_epsilon};"
        else:
            u_init = np.zeros((A, self.nx, self.ny))

        f = equilibrium(rho_init, u_init)

        for idx, value in enumerate(omega_list):
            print(f"Omega value: {value}")
            self.omega = value
            _, _, u_amplitude, r_amplitude = self.simulate_shear_wave_decay(f, plot=False)
            if "density" in self.config_title:
                a_0 = abs(r_amplitude[0]-self.rho_o)
                a_1 = abs(r_amplitude[1]-self.rho_o)
                k = (2.*np.pi)/self.L
                visc_plot = (np.log(a_0)-np.log(a_1))/(k*k*self.nt_log)
            elif "velocity" in self.config_title:
                a_0 = u_amplitude[0]
                a_1 = u_amplitude[1]
                k = (2.*np.pi)/self.L
                visc_plot = (np.log(a_0)-np.log(a_1))/(k*k*self.nt_log)
            
            visc_calc = (1/3)*((1/self.omega)-0.5)
            viscosities[idx][:] = self.omega, visc_calc, visc_plot 

        viscosities = viscosities.T
        plt.plot(viscosities[0], viscosities[1], 'o-', label="Analytical")
        plt.plot(viscosities[0], viscosities[2], 'o-', label="Simulation plot")
        plt.xlabel("Omega")
        plt.ylabel("Viscosity")
        plt.title(f"Comparison of omega - Simulation plot vs Formula -{self.config_title}", wrap=True)
        plt.legend()
        plt.show(block=False)
        os.makedirs(f"{self.path}", exist_ok=True)
        plt.savefig(f"{self.path}/Omega_comparison_{self.config_title.split('-')[0].strip()}_{self.omega_min}_{self.omega_max}_{self.omega_count}.png")
        plt.clf()
        plt.cla()
        plt.close()
    
    def run(self):
        """
        Run Shear Wave Decay simulation and plot additional inference plots.
        """
        self.config_title = f"Omega-{self.omega};"

        if self.rho_o is not None and self.rho_epsilon is not None:
            rho_init = self.rho_o + self.rho_epsilon*np.sin((2.*np.pi*self.X)/self.L)
            self.config_title += \
            f" Sinusoidal density - rho_o-{self.rho_o};rho_epsilon-{self.rho_epsilon};"
        else:
            rho_init = np.ones((self.nx, self.ny))

        if self.u_epsilon is not None:
            ux = self.u_epsilon*np.sin((2.*np.pi*self.X)/self.L)
            uy = np.zeros((self.nx, self.ny))
            u_init = np.stack((ux,uy), axis = 0)
            self.config_title += f" Sinusoidal velocity - u_epsilon-{self.u_epsilon};"
        else:
            u_init = np.zeros((A, self.nx, self.ny))

        f = equilibrium(rho_init, u_init)

        os.makedirs(f"{self.path}/{self.config_title}", exist_ok=True)
        
        u_periodic, r_periodic, u_amplitude, r_amplitude = \
        self.simulate_shear_wave_decay(f)

        # Wave decay
        plot_decay(r_periodic, "Y-coordinate", "Density", "Density variation plot - "+self.config_title, rho_init[self.nx//2], f"{self.path}/{self.config_title}", nt_log=self.nt_log)

        # Amplitude plot
        plot_amplitude(r_amplitude, f"Timestep (in {self.nt_log}) ", "Amplitude of density", "Density amplitude plot - "+self.config_title, f"{self.path}/{self.config_title}")

        # Combined decay plot
        plot_combined(r_periodic,
                      "Y-coordinate", "Density",
                      f"{self.path}/{self.config_title}",
                      title=f"Density_combined_plot - {self.config_title}")

        # Wave decay
        plot_decay(u_periodic, "Y-coordinate", "Velocity", "Velocity variation plot - "+self.config_title, u_init[0,self.nx//2], f"{self.path}/{self.config_title}", nt_log=self.nt_log)

        # Amplitude plot
        plot_amplitude(u_amplitude, f"Timestep (in {self.nt_log}) ", "Amplitude of velocity", "Velocity amplitude plot - "+self.config_title, f"{self.path}/{self.config_title}")

        # Combined decay plot
        plot_combined(u_periodic,
                      "Y-coordinate", "Velocity",
                      f"{self.path}/{self.config_title}",
                      title=f"Velocity_combined_plot - {self.config_title}")

if __name__ == "__main__":
    swd = ShearWaveDecay()
    if swd.omega_comparison:
        swd.run_multiple_omega()
    else:
        swd.run()