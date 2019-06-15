import h5py
import numpy as np
import matplotlib.pyplot as plt
from init_transportation import *
import csv


if __name__ == "__main__":
    beamline = U3_no_kick
    
    Matrix_List = np.zeros((4,4,len(beamline)))

    for n in range(len(beamline)):
        Matrix_List[:,:,n] = beamline[n].M1

    np.save("Matrix_List", Matrix_List)


