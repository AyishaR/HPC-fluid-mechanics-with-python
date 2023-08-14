"""
Boundary conditions are defined. Stationary and moving boundary conditions are toggled by a boolean parameter.

Moving boundary is implemented only for the top boundary (as other cases were not required). Can be extended for other boundaries too.
"""

def top_boundary(f, moving=False, ub=0, vb=0):
    f[4,1:-1,-2] = f[2,1:-1,-1]
    f[7,1:-1,-2] = f[5,2:,-1]
    f[8,1:-1,-2] = f[6,0:-2,-1]
    
    if moving:
        rho_n = (f[0,1:-1,-2] + f[1,1:-1,-2] + f[3,1:-1,-2] + 
                     2*(f[2,1:-1,-1] + f[6,0:-2,-1] + f[5,2:,-1]))/(1+vb)
        horr_factor = 0.5*(f[1,1:-1,-2]-f[3,1:-1,-2])
        ub_factor = 0.5*rho_n*ub
        vb_factor = (1/6)*rho_n*vb
        f[4,1:-1,-2] += (- (2/3)*rho_n*vb)
        f[7,1:-1,-2] += (horr_factor - ub_factor - vb_factor)
        f[8,1:-1,-2] += (- horr_factor + ub_factor - vb_factor)
        
    return f

def bottom_boundary(f, moving=False, ub=0, vb=0):
    f[2,1:-1,1] = f[4,1:-1,0]
    f[5,1:-1,1] = f[7,0:-2,0]
    f[6,1:-1,1] = f[8,2:,0]
    
    if moving:
        raise NotImplementedError()
        
    return f

def left_boundary(f, moving=False, ub=0, vb=0):
    f[1,1,1:-1] = f[3,0,1:-1]
    f[8,1,1:-1] = f[6,0,2:]
    f[5,1,1:-1] = f[7,0,0:-2]
    
    if moving:
        raise NotImplementedError()
        
    return f

def right_boundary(f, moving=False, ub=0, vb=0):
    f[3,-2,1:-1] = f[1,-1,1:-1]
    f[7,-2,1:-1] = f[5,-1,2:]
    f[6,-2,1:-1] = f[8,-1,0:-2]
    
    if moving:
        raise NotImplementedError()
        
    return f
    