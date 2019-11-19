import numpy as np 
import matplotlib.pyplot as plt 
import h5py 
from LCLS_beamline_exact import *
from twiss_matrix_transportation import *

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

    lattice = U1_core[1:25]

    for element in lattice:
        M = np.dot(element.M1, M)

    print(M)

    # First, I need to calculate the Twiss parameter at the start of the first periodic structure.

    alpha_x, beta_x, gamma_x = calculate_twiss(M[0:2, 0:2])
    alpha_y, beta_y, gamma_y = calculate_twiss(M[2:4, 2:4])

    # However, in our beamline there is a small drift before the start of the periodic structure.
    # In order to calculate the accurate machine optics, we need to back propagate the Twiss 
    # calculated above to the start of the undulator beamline.

    Twiss_at_UND01 = np.array([alpha_x, beta_x, gamma_x, alpha_y, beta_y, gamma_y]).reshape((6,1))

    M_D_init = U1_core[0].M1

    Twiss_at_D_init = twiss_after_transportation(np.linalg.inv(M_D_init), Twiss_at_UND01)

    print('Twiss parameters in X:\n')
    print(Twiss_at_D_init[0:3,0])
    print('\n')
    print('Twiss parameters in Y:\n')
    print(Twiss_at_D_init[3:6,0])
