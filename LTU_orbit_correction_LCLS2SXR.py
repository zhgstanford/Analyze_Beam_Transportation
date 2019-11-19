import h5py
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from init_transportation import *

def LTU_orbit(ps_slice, id_slice):
    """We want ot use this function to calculate the kick in XCUM1, YCUM2, YCUM3, XCUM4.
    """

    ps0 = ps_slice[0:4, id_slice]

    M = np.array([[3.862342331536209e-01, 1.319971108468322e+01, 0, 0],
            [-2.895398044999400e-02, 1.599588462845505e+00, 0, 0],
            [0, 0, 1.612992595330453e+00, 1.320027250295609e+01],
            [0, 0, -2.692475335037540e-02,  3.996211269451667e-01]])

    M_equations = np.zeros((4,4))
    
    M_equations[0:4, 0] = M[0:4, 1]
    M_equations[0:4, 1] = M[0:4, 3]
    M_equations[0:4, 2] = np.array([0,1,0,0]).T
    M_equations[0:4, 3] = np.array([0,0,0,1]).T

    M_kick = -np.matmul(M, ps0)
    kick = np.matmul(inv(M_equations), M_kick)

    print('XCSX21, YCSX21, XCSX24, YCSX24')
    print(kick)

    XCSX21 = kick[0] 
    YCSX21 = kick[1]
    XCSX24 = kick[2]
    YCSX24 = kick[3]


    return XCSX21, YCSX21, XCSX24, YCSX24

if __name__ == "__main__":

    del bunch 
    del gamma

    bunch = h5py.File('/home/zhaohengguo/Desktop/GENESIS4_Input/LCLS2SXR_DECHin_no_OC.bun', 'r')
    gamma = bunch['pCentral'][()] # The central energy of the bunch

    N_bin = 40

    ps_beg = np.zeros((4, len(bunch["t"])))

    #plt.hist(np.ravel(bunch["t"]), bins = N_bin)
    #plt.show()

    ds_len = np.ptp(bunch["t"])/N_bin # The length of one slice in time

    ps_beg[0, :] = bunch["x"]
    ps_beg[1, :] = bunch["xp"]
    ps_beg[2, :] = bunch["y"]
    ps_beg[3, :] = bunch["yp"]

    id_slices, zplot, hist = flip_slice(bunch["t"], bins = N_bin)
    ps_slice = beam_property_along_s(ps_beg, id_slices)

    # Remember: 0 is the tail of the beam
    XCUM1, YCUM2, YCUM3, XCUM4 = LTU_orbit(ps_slice, 16)




