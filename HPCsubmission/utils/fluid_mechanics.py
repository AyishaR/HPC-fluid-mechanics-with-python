"""
Functions for basic fluid mechanics entities such as density, velocity and events such as streaming, collision and others are defined.
"""

import numpy as np
from utils.constants import *

def get_rho(f):
    """
    Calculate local density from individual density values in the lattice grid.

    :param f: Lattice grid probability density
    :type f: np.array
    :return: Local density of the lattice grid
    :rtype: np.array
    """
    return np.einsum("ijk->jk", f)

def get_u(f, rho):
    """
    Calculate the velocity (along both axes) of the lattice grid.

    :param f: Lattice grid probability density
    :type f: np.array
    :param rho: Density of the lattice grid
    :type rho: np.array
    :return: Velocity of the lattice grid
    :rtype: np.array
    """
    return np.einsum('ai,ijk->ajk',C_AI,f)/rho

def stream(f):
    """
    Stream the lattice grid for one time step along the channels.

    :param f: Lattice grid probability density
    :type f: np.array  
    :return: Lattice grid probability density after streaming
    :rtype: np.array
    """
    for i in range(9):
        f[i] = np.roll(f[i], shift = C_AI.T[i], axis=[0,1])
    return f

def equilibrium(rho, u):
    """
    Calculate the equilibrium state of the lattice based on the density and the velocity.

    :param rho: Density of the lattice grid
    :type rho: np.array
    :param u: Velocity of the lattice grid
    :type u: np.array
    :return: Equilibrium state of the lattice grid
    :rtype: np.array
    """
    feq = []
    for i in range(9):
        cu = np.einsum('a,anm->nm', C_AI.T[i], u)
        uu = np.einsum('anm,anm->nm',u,u)
        term = W_I[i]*rho*(1 + (3 * cu) + (4.5 * (cu**2)) - (1.5 * uu))
        feq.append(term)
    feq = np.stack(feq)
    return feq

def equilibrium_i(rho, u):
    """
    Calculate the equilibrium state of one row or column of the lattice based on the density and the velocity.

    :param rho: Density of the lattice grid
    :type rho: np.array
    :param u: Velocity of the lattice grid
    :type u: np.array
    :return: Equilibrium state of the lattice grid
    :rtype: np.array
    """
    feq = []
    for i in range(9):
        cu = np.einsum('a,an->n', C_AI.T[i], u)
        uu = np.einsum('an,an->n',u,u)
        term = W_I[i]*rho*(1 + (3 * cu) + (4.5 * (cu**2)) - (1.5 * uu))
        feq.append(term)
    feq = np.stack(feq)
    return feq

def collision(f, omega):
    """
    Simulate collision using the equilibrium state of the lattice

    :param f: Lattice grid probability density
    :type f: np.array
    :param omega: Omega value for the simulation
    :type omega: float
    :return: Lattice grid probability density after collision
    :rtype: np.array
    """
    rho = get_rho(f)
    u = get_u(f,rho)
    feq = equilibrium(rho, u)
    
    f = f + (omega * (feq - f))
    return f

def rho_in(n, rho_in_value=1,):
    """
    Calculate density along hte inlet of the pressure pipe. 

    :param n: Size of the inlet of the pressure pipe
    :type n: int
    :param rho_in_value: Value of inlet density, defaults to 1
    :type rho_in_value: float, optional
    :return: Density at inlet of the pressure pipe
    :rtype: np.array
    """
    f = np.einsum("i,j->ij", W_I, np.ones((n+2))*rho_in_value)
    rho = np.einsum("ij->j", f)
    return rho

def rho_out(n, rho_out_value=1):
    """
    Calculate density along the inlet of the pressure pipe. 

    :param n: Size of the inlet of the pressure pipe
    :type n: int
    :param rho_in_value: Value of inlet density, defaults to 1
    :type rho_in_value: float, optional
    :return: Density at inlet of the pressure pipe
    :rtype: np.array
    """
    f = np.einsum("i,j->ij", W_I, np.ones((n+2))*rho_out_value)
    rho = np.einsum("ij->j", f)
    return rho

def set_pressure_gradient(f, n, rho_in_value=1, rho_out_value=1):
    """
    Update lattice grid to apply pressure gradient using a difference in density between inlet and outlet.

    :param f: Lattice grid probability density
    :type f: np.array
    :param n: Size of the inlet/outlet of the pressure pipe
    :type n: int
    :param rho_in_value: Density at inlet of the pressure pipe, defaults to 1
    :type rho_in_value: float, optional
    :param rho_out_value: Density at outlet of the pressure pipe, defaults to 1
    :type rho_out_value: float, optional
    :return: Lattice grid probability density with pressure gradient applied
    :rtype: np.array
    """
    rho_p = get_rho(f)
    u_p = get_u(f, rho_p)
    
    feq_star = equilibrium(rho_p,u_p)
    f[:,0,:] = equilibrium_i(rho_in(n, rho_in_value), u_p[:,-2,:]) + (f[:,-2,:] - feq_star[:,-2,:])
    f[:,-1,:] = equilibrium_i(rho_out(n, rho_out_value), u_p[:,1,:]) + (f[:,1,:] - feq_star[:,1,:])
        
    return f