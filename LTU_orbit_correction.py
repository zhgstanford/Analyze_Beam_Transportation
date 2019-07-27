import h5py
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from LCLS_beamline import *

def LTU_orbit(ps_slice, id_slice, XCUM1_to_YCUM2, YCUM2_to_YCUM3, YCUM3_to_XCUM4):
    """We want ot use this function to calculate the kick in XCUM1, YCUM2, YCUM3, XCUM4.
    """

    ps0 = ps_slice[0:4, id_slice]


    M1 = np.identity(4)
    M2 = np.identity(4)
    M3 = np.identity(4)

    for element in XCUM1_to_YCUM2:
        M1 = np.dot(element.M1, M1)

    for element in YCUM2_to_YCUM3:
        M2 = np.dot(element.M1, M2)

    for element in YCUM3_to_XCUM4:
        M3 = np.dot(element.M1, M3)

    M_equations = np.zeros((4,4))
    
    M_equations[0:4, 0] = np.matmul(M3, np.matmul(M2, M1))[0:4, 1]
    M_equations[0:4, 1] = np.matmul(M3, M2)[0:4, 3]
    M_equations[0:4, 2] = M3[0:4, 3]
    M_equations[0:4, 3] = np.array([0,1,0,0]).T

    M_kick = -np.matmul(M3, np.matmul(M2, np.matmul(M1, ps0)))
    kick = np.matmul(inv(M_equations), M_kick)

    print('XCUM1, YCUM2, YCUM3, XCUM4')
    print(kick)

    XCUM1 = kick[0] 
    YCUM2 = kick[1]
    YCUM3 = kick[2]
    XCUM4 = kick[3]


    return XCUM1, YCUM2, YCUM3, XCUM4

if __name__ == "__main__":

    bunch = h5py.File('/home/zhaohengguo/Desktop/GENESIS4_Input/X15Y14_noOC.bun', 'r')
    gamma = bunch['pCentral'][()] # The central energy of the bunch

    N_bin = 500
    lambdaref = 1.84631767e-09 # Central XFEL wavelength

    ps_beg = np.zeros((4, len(bunch["t"])))

    ps_beg[0, :] = bunch["x"]
    ps_beg[1, :] = bunch["xp"]
    ps_beg[2, :] = bunch["y"]
    ps_beg[3, :] = bunch["yp"]

    id_slices, zplot, hist = flip_slice(bunch["t"], bins = N_bin)
    ps_slice = beam_property_along_s(ps_beg, id_slices)

    # Remember: 0 is the tail of the beam
    XCUM1, YCUM2, YCUM3, XCUM4 = LTU_orbit(ps_slice, 80, XCUM1_to_YCUM2, YCUM2_to_YCUM3, YCUM3_to_XCUM4)


    ps_end = beam_transportation(ps_beg, XCUM1_to_XCUM4)
    ps_end_slice = beam_property_along_s(ps_end, id_slices)[0:4,:]

    plt.plot(ps_end_slice[0,:])
    plt.plot(ps_end_slice[1,:])
    plt.plot(ps_end_slice[2,:])
    plt.plot(ps_end_slice[3,:])
    plt.show()




