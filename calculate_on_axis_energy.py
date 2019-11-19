import numpy as np 
import matplotlib.pyplot as plt 
import h5py

output = h5py.File('/home/zhaohengguo/Desktop/Codes_from_Alberto/LCLS2_SXR_Two_Stage.out.h5', 'r')

lambdaref_1 = 2.33932e-9 # m
lambdaref_2 = 2.43106e-9
lambuda_u = 0.039 # m

y = output["Beam/yposition"] ##(n_z, n_s)
x = output["Beam/xposition"] ##(n_z, n_s)

ysize = output["Beam/ysize"] ##(n_z, n_s)
xsize = output["Beam/xsize"] ##(n_z, n_s)

energy = output["Beam/energy"] ##(n_z, n_s)
ku = output["Lattice/ku"] 

# Some part of the beam hasn't been calculated at all. We need to discard this meaningless part.
cutoff = 1829

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

U1_on_axis_id = np.argmax(np.bincount(on_axis_id[0:OC_id[0]]))
x_size_U1 = xsize[0:OC_id[0], U1_on_axis_id]
y_size_U1 = ysize[0:OC_id[0], U1_on_axis_id]
E_axis_U1 = energy[0:OC_id[0], U1_on_axis_id]

U2_on_axis_id = np.argmax(np.bincount(on_axis_id[OC_id[1]:-1]))
x_size_U2 = xsize[OC_id[1]:-1, U2_on_axis_id]
y_size_U2 = ysize[OC_id[1]:-1, U2_on_axis_id]
E_axis_U2 = energy[OC_id[1]:-1, U2_on_axis_id]

# delay1 = (U2_on_axis_id-U1_on_axis_id)*lambdaref*5 - 8*110*lambdaref
# delay2 = (U3_on_axis_id-U2_on_axis_id)*lambdaref*5 - 6*110*lambdaref


## Calculate the desired taper in all three sections along z

K_U1 = np.sqrt(4*E_axis_U1**2*lambdaref_1/lambuda_u-2)
aw_U1 = K_U1/np.sqrt(2)

K_U2 = np.sqrt(4*E_axis_U2**2*lambdaref_2/lambuda_u-2)
aw_U2 = K_U2/np.sqrt(2)



#####################################################################
#####################   Plot Energy Change   ########################
#####################################################################

# plt.figure()
# plt.plot(E_axis_U1)
# plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
# plt.title('First Undulator Section')
# plt.show()

# plt.figure()
# plt.plot(E_axis_U2)
# plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
# plt.title('Second Undulator Section')
# plt.show()
