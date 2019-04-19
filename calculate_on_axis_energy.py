import numpy as np 
import matplotlib.pyplot as plt 
import h5py

output = h5py.File('/Users/guozhaoheng/Desktop/Genesis4/GENESIS_output_file/test20/test20.out.h5', 'r')

lambdaref = output["Global/lambdaref"][()] # m
lambuda_u = 0.03 # m

y = output["Beam/yposition"] ##(n_z, n_s)
x = output["Beam/xposition"] ##(n_z, n_s)

ysize = output["Beam/ysize"] ##(n_z, n_s)
xsize = output["Beam/xsize"] ##(n_z, n_s)

energy = output["Beam/energy"] ##(n_z, n_s)
ku = output["Lattice/ku"] 

# Some part of the beam hasn't been calculated at all. We need to discard this meaningless part.
cutoff = np.where(np.abs(y[0,:])<1e-20)[0][0]-1

y = y[:, 0:cutoff]
x = x[:, 0:cutoff]
ysize = ysize[:, 0:cutoff]
xsize = xsize[:, 0:cutoff]
on_axis_id = np.argmin(np.power(x, 2)+np.power(y, 2), axis =1)

energy = energy[:, 0:cutoff]

# Indices of slices in z where all undulators begin
UND_id = np.where(np.diff(ku)>0)[0]+1 


# Use orbit correctors to find all three undulator sections
cx = output["Lattice/cx"] ## (n_z, 1)
OC_id = np.ravel(np.nonzero(cx)) ## Usually this list has 6 elements since we have 3 pairs of orbit correctors.

U1_on_axis_id = np.argmax(np.bincount(on_axis_id[OC_id[1]:OC_id[2]]))
x_size_U1 = xsize[OC_id[1]:OC_id[2], U1_on_axis_id]
y_size_U1 = ysize[OC_id[1]:OC_id[2], U1_on_axis_id]
E_axis_U1 = energy[OC_id[1]:OC_id[2], U1_on_axis_id]

U2_on_axis_id = np.argmax(np.bincount(on_axis_id[OC_id[3]:OC_id[4]]))
x_size_U2 = xsize[OC_id[3]:OC_id[4], U2_on_axis_id]
y_size_U2 = ysize[OC_id[3]:OC_id[4], U2_on_axis_id]
E_axis_U2 = energy[OC_id[3]:OC_id[4], U2_on_axis_id]

U3_on_axis_id = np.argmax(np.bincount(on_axis_id[OC_id[5]:-1]))
x_size_U3 = xsize[OC_id[5]:(len(cx)-1), U3_on_axis_id]
y_size_U3 = ysize[OC_id[5]:(len(cx)-1), U3_on_axis_id]
E_axis_U3 = energy[OC_id[5]:(len(cx)-1), U3_on_axis_id]

# delay1 = (U2_on_axis_id-U1_on_axis_id)*lambdaref*5 - 8*110*lambdaref
# delay2 = (U3_on_axis_id-U2_on_axis_id)*lambdaref*5 - 6*110*lambdaref


## Calculate the desired taper in all three sections along z

K_U1 = np.sqrt(4*E_axis_U1**2*lambdaref/lambuda_u-2)
aw_U1 = K_U1/np.sqrt(2)

K_U2 = np.sqrt(4*E_axis_U2**2*lambdaref/lambuda_u-2)
aw_U2 = K_U2/np.sqrt(2)

K_U3 = np.sqrt(4*E_axis_U3**2*lambdaref/lambuda_u-2)
aw_U3 = K_U3/np.sqrt(2)

## Calculate the desired taper in all undulators
taper_U1 = aw_U1[UND_id[1:8]-OC_id[1]]
taper_U2 = aw_U2[UND_id[8:14]-OC_id[3]]
taper_U3 = aw_U3[UND_id[14:]-OC_id[5]]

#####################################################################
#####################  Plot Twiss Parameters ########################
#####################################################################

# plt.figure()
# plt.plot(y_size_U1*1e6)
# plt.ylabel('Y Size/ $\mu$m')
# plt.title('First Undulator Section')
# plt.savefig('ysize_U1.png')

# plt.figure()
# plt.plot(x_size_U1*1e6)
# plt.ylabel('X Size/ $\mu$m')
# plt.title('First Undulator Section')
# plt.savefig('xsize_U1.png')

# plt.figure()
# plt.plot(y_size_U2*1e6)
# plt.ylabel('Y Size/ $\mu$m')
# plt.title('Second Undulator Section')
# plt.savefig('ysize_U2.png')

# plt.figure()
# plt.plot(x_size_U2*1e6)
# plt.ylabel('X Size/ $\mu$m')
# plt.title('Second Undulator Section')
# plt.savefig('xsize_U2.png')

# plt.figure()
# plt.plot(y_size_U3*1e6)
# plt.ylabel('Y Size/ $\mu$m')
# plt.title('Third Undulator Section')
# plt.savefig('ysize_U3.png')

# plt.figure()
# plt.plot(x_size_U3*1e6)
# plt.ylabel('X Size/ $\mu$m')
# plt.title('Third Undulator Section')
# plt.savefig('xsize_U3.png')

#####################################################################
#####################   Plot Energy Change   ########################
#####################################################################

plt.figure()
plt.plot(E_axis_U1)
plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
plt.title('First Undulator Section')
plt.savefig('gamma_U1.png')

plt.figure()
plt.plot(E_axis_U2)
plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
plt.title('Second Undulator Section')
plt.savefig('gamma_U2.png')

plt.figure()
plt.plot(E_axis_U3)
plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
plt.title('Third Undulator Section')
plt.savefig('gamma_U3.png')