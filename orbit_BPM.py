# I want to use this script to calculate the center-of-mass orbit at the end of each quadrupole

import numpy as np 
import matplotlib.pyplot as plt 
import h5py

output = h5py.File('/home/zhaohengguo/Desktop/GENESIS4_Output/scan_taper_whole_U3.out.h5', 'r')

lambdaref = output["Global/lambdaref"][()] # m
lambuda_u = 0.03 # m

y = output["Beam/yposition"] ##(n_z, n_s)
x = output["Beam/xposition"] ##(n_z, n_s)


ysize = output["Beam/ysize"] ##(n_z, n_s)
xsize = output["Beam/xsize"] ##(n_z, n_s)
qf = output["Lattice/qf"] ##(n_z, 1)

cutoff = np.where(np.abs(y[0,:])<1e-20)[0][0]-1
#cutoff = np.size(y, -1)

y = y[:, 0:cutoff]
x = x[:, 0:cutoff]
ysize = ysize[:, 0:cutoff]
xsize = xsize[:, 0:cutoff]

BPM_id = np.where(np.diff(np.abs(qf))<0)[0]+1

x_whole_BPM = x[BPM_id, :]
y_whole_BPM = y[BPM_id, :]

x_BPM = np.average(x[BPM_id, :], axis=1)
y_BPM = np.average(y[BPM_id, :], axis=1)

plt.figure()
plt.plot(x_BPM*1e6)
plt.ylabel('X Orbit/$\mu$m')
plt.xlabel('Id of quadrupole')
plt.title('X Orbit at the End of Each Quadrupole')
plt.savefig('x_BPM.png')

plt.figure()
plt.plot(y_BPM*1e6)
plt.ylabel('Y Orbit/$\mu$m')
plt.xlabel('Id of quadrupole')
plt.title('Y Orbit at the End of Each Quadrupole')
plt.savefig('y_BPM.png')
