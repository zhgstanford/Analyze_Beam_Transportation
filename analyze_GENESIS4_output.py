import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    output = h5py.File('/home/zhaohengguo/Desktop/Analyze_Beam_Transportation/GENESIS4_Output/test0116_1.h5', 'r')

    ds = 5*output["Global/lambdaref"][()]
    current = np.ravel(output["Beam/current"])

    xsize = output["Beam/xsize"]
    ysize = output["Beam/ysize"]

    xposition = output["Beam/xposition"]
    yposition = output["Beam/yposition"]
    transverse_p = np.sqrt(np.power(xposition, 2)+np.power(yposition, 2))

    on_axis_id = np.argmin(transverse_p, axis = 1)

    U1_slice = np.ceil(np.average(on_axis_id[200:500]))
    U2_slice = np.ceil(np.average(on_axis_id[750:1100]))
    U3_slice = np.ceil(np.average(on_axis_id[1300:1700]))

    print(U1_slice)
    print(U2_slice)
    print(U3_slice)
