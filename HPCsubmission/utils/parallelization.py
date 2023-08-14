"""
Functions for parallel implementation and execution of the module.
"""

from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from utils.constants import *

class Parallelization:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.allrcoords = None
        self.sectsX, self.sectsY = None, None
        self.nxsub, self.nysub = None, None
        self.boundary_k, self.cartcomm, self.rcoords, self.sd = None, None, None, None

        self.nx, self.ny = None, None

    def grid_division(self):

        self.sectsX=int(np.floor(np.sqrt(self.size)))
        self.sectsY=int(self.size//self.sectsX)
        if self.rank == 0: 
            print('The processes are divided as {} and {}'.format(self.sectsX,self.sectsY))

        self.nxsub = self.nx//self.sectsX+2
        self.nysub = self.ny//self.sectsY+2
        self.boundary_k = [False,False,False,False]
        self.cartcomm = self.comm.Create_cart(dims=[self.sectsX,self.sectsY],periods=[True,True],reorder=False)
        self.rcoords = self.cartcomm.Get_coords(self.rank)
        self.allrcoords = self.comm.gather(self.rcoords, root = 0)

        sR,dR = self.cartcomm.Shift(0,1)
        sL,dL = self.cartcomm.Shift(0,-1)
        sU,dU = self.cartcomm.Shift(1,1)
        sD,dD = self.cartcomm.Shift(1,-1)

        self.sd = np.array([sR,dR,sL,dL,sU,dU,sD,dD], dtype = int)

    # Communication between subdomains
    def communicate(self, c):
        sR,dR,sL,dL,sU,dU,sD,dD = self.sd
        
        recvbuf = np.zeros(c[:,1,:].shape)

        sendbuf = c[:,-2,:].copy()
        self.cartcomm.Sendrecv(sendbuf, dR, recvbuf = recvbuf, source = sR)
        c[:,0,:] = recvbuf 

        sendbuf = c[:,1,:].copy()
        self.cartcomm.Sendrecv(sendbuf, dL, recvbuf = recvbuf, source = sL)
        c[:,-1,:] = recvbuf

        recvbuf = np.zeros(c[:,:,1].shape)
        
        sendbuf = c[:,:,1].copy()
        self.cartcomm.Sendrecv(sendbuf, dD, recvbuf = recvbuf, source = sD)
        c[:,:,-1] = recvbuf

        sendbuf = c[:,:,-2].copy()
        self.cartcomm.Sendrecv(sendbuf, dU, recvbuf = recvbuf, source = sU)
        c[:,:,0]=recvbuf

        return c

    def get_coordinates_of_rank(self, r=None):
        if r:
            rc = self.allrcoords[r]
        else:
            rc = self.rcoords
        x_grid = self.nx//self.sectsX
        y_grid = self.ny//self.sectsY
        xlo = x_grid*rc[0]
        xhi = x_grid*(rc[0]+1)
        ylo = y_grid*rc[1]
        yhi = y_grid*(rc[1]+1)
        return xlo, xhi, ylo, yhi
    
    def get_boundaries(self):
        b = []
        if self.rcoords[0]==0: 
            b.append(LEFT)
        if self.rcoords[0]==self.sectsX-1:
            b.append(RIGHT)
        if self.rcoords[1]==0: 
            b.append(BOTTOM)
        if self.rcoords[1]==self.sectsY-1:
            b.append(TOP)
        return b