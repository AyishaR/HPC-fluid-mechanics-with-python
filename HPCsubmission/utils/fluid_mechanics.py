"""
Functions for basic fluid mechanics entities such as density, velocity and events such as streaming, collision and others are defined.
"""

import numpy as np
from utils.constants import *

def get_rho(f):
    return np.einsum("ijk->jk", f)

def get_u(f, rho):
    return np.einsum('ai,ijk->ajk',C_AI,f)/rho

def stream(f):
    for i in range(9):
        f[i] = np.roll(f[i], shift = C_AI.T[i], axis=[0,1])
    return f

def equilibrium(rho, u):
    feq = []
    for i in range(9):
        cu = np.einsum('a,anm->nm', C_AI.T[i], u)
        uu = np.einsum('anm,anm->nm',u,u)
        term = W_I[i]*rho*(1 + (3 * cu) + (4.5 * (cu**2)) - (1.5 * uu))
        feq.append(term)
    feq = np.stack(feq)
    return feq

def equilibrium_i(rho, u):
    feq = []
    for i in range(9):
        cu = np.einsum('a,an->n', C_AI.T[i], u)
        uu = np.einsum('an,an->n',u,u)
        term = W_I[i]*rho*(1 + (3 * cu) + (4.5 * (cu**2)) - (1.5 * uu))
        feq.append(term)
    feq = np.stack(feq)
    return feq

def collision(f, omega):
    rho = get_rho(f)
    u = get_u(f,rho)
    feq = equilibrium(rho, u)
    
    f = f + (omega * (feq - f))
    return f

def rho_in(n, rho_in_value=1,):
    f = np.einsum("i,j->ij", W_I, np.ones((n+2))*rho_in_value)
    rho = np.einsum("ij->j", f)
    return rho

def rho_out(n, rho_out_value=1):
    f = np.einsum("i,j->ij", W_I, np.ones((n+2))*rho_out_value)
    rho = np.einsum("ij->j", f)
    return rho

def set_pressure_gradient(f, n, rho_in_value=1, rho_out_value=1):
    rho_p = get_rho(f)
    u_p = get_u(f, rho_p)
    
    feq_star = equilibrium(rho_p,u_p)
    f[:,0,:] = equilibrium_i(rho_in(n, rho_in_value), u_p[:,-2,:]) + (f[:,-2,:] - feq_star[:,-2,:])
    f[:,-1,:] = equilibrium_i(rho_out(n, rho_out_value), u_p[:,1,:]) + (f[:,1,:] - feq_star[:,1,:])
        
    return f