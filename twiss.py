import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
from init_transportation import *

def calculate_twiss(M):
    """Input the 2D transform matrix and output Twiss parameters in X and Y."""
    
    if M[0,1]>0:
        ## Then we have 0 < phi < pi.
        phi = np.arccos(0.5*np.matrix.trace(M))
    else:
        phi = np.arccos(0.5*np.matrix.trace(M))+np.pi

    alpha = (M[0,0]-M[1,1])/(2*np.sin(phi))
    beta = M[0,1]/np.sin(phi)
    gamma = -M[1,0]/np.sin(phi)

    print(phi)

    return alpha, beta, gamma


if __name__ == "__main__":
    M = np.identity(4)

    lattice = U3[1:9]

    for element in lattice:
        M = np.dot(element.M1, M)

    print(M)

    alpha_x, beta_x, gamma_x = calculate_twiss(M[0:2, 0:2])
    alpha_y, beta_y, gamma_y = calculate_twiss(M[2:4, 2:4])
    print('Twiss parameters in X:\n')
    print(alpha_x, beta_x, gamma_x)
    print('\n')
    print('Twiss parameters in Y:\n')
    print(alpha_y, beta_y, gamma_y)
