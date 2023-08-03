import math
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.plots import *
from utils.fluid_mechanics import *
from utils.constants import *

class ShearWaveDecay:
    def __init__(self) -> None:
        self.omega = 1.2

        self.rho_o = None
        self.rho_epsilon = None
        self.u_epsilon = 0.01

        self.omega_min = 0.1
        self.omega_max = 1.5
        self.omega_count = 10

        self.nx = 20
        self.ny = 20
        self.L = self.ny
        self.X, self.Y = np.meshgrid(np.arange(0,self.nx), np.arange(0,self.ny))

        self.subplot_columns = 5
        self.nt = 100
        self.nt_log = 10

        self.config_title = ""
        self.path = f"plots/ShearWaveDecay/{self.config_title}"

    def shear_wave_decay(self, f_inm):
        # Streaming
        f_inm = stream(f_inm)
        
        # Collision
        f_inm = collision(f_inm, self.omega)
    
        return f_inm
    
    def simulate_shear_wave_decay(self,
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
            plt.setp(axes, xticks=range(0,self.nx+1,5), yticks=range(0,self.ny+1,5))
            plt.gcf().set_size_inches(self.subplot_columns*3,subplot_rows*3)
            if self.config_title:
                plt.suptitle(self.config_title)

        for i in range(self.nt):
            f = self.shear_wave_decay(f)
            if i%self.nt_log==0:
                print(i)
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
            plt.savefig(f"{self.path}/{self.config_title}/Streamplot.png")
        return u_periodic, r_periodic, u_amplitude, r_amplitude
    
    def run_multiple_omega(self):
        # TODO: Move to command line args
        omega_list = np.round(np.linspace(self.omega_min,
                                          self.omega_max,
                                          self.omega_count)[1:],2)
        viscosities = np.zeros((len(omega_list), 3))


        for idx, value in enumerate(omega_list):
            print(f"Omega value: {value}")
            self.omega = value
            rho_init = np.ones((self.nx, self.ny))
            u_init = np.stack((self.u_epsilon*np.sin((2.*np.pi*self.X)/self.L),
                            np.zeros((self.nx,self.ny))), 
                            axis = 0)
            f = equilibrium(rho_init, u_init)
            _, _, u_amplitude, _ = self.simulate_shear_wave_decay(f, plot=False)
            
            a_0 = u_amplitude[0]
            a_1 = u_amplitude[1]
            k = (2.*np.pi)/self.L
            visc_plot = (np.log(a_0)-np.log(a_1))/(k*k*self.nt_log)
            
            visc_calc = (1/3)*((1/self.omega)-0.5)
            viscosities[idx][:] = self.omega, visc_calc, visc_plot 

        viscosities = viscosities.T
        plt.plot(viscosities[0], viscosities[1], label="Formula")
        plt.plot(viscosities[0], viscosities[2], label="Simulation plot")
        plt.xlabel("Omega")
        plt.ylabel("Viscosity")
        plt.title("Comparison of omega - Simulation plot vs Formula")
        plt.legend()
        plt.show()
        plt.savefig(f"{self.path}/Omega_comparison_{self.omega_min}_{self.omega_max}_{self.omega_count}.png")
    
    def run(self):
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
        plot_decay(r_periodic, "Y-coordinate", "Density", "Density variation plot - "+self.config_title, rho_init[self.nx//2], f"{self.path}/{self.config_title}")

        # Amplitude plot
        plot_amplitude(r_amplitude, f"Timestep (in {self.nt_log}) ", "Amplitude of density", "Density amplitude plot - "+self.config_title, f"{self.path}/{self.config_title}")

swd = ShearWaveDecay()
# swd.run()
swd.run_multiple_omega()