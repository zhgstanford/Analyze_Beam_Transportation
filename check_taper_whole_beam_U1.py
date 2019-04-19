import numpy as np 
import matplotlib.pyplot as plt 
import h5py

output = h5py.File('/home/zhaohengguo/Desktop/Analyze_Beam_Transportation/GENESIS4_Output/scan_taper_U1_8.out.h5', 'r')

lambdaref = output["Global/lambdaref"][()] # m
lambuda_u = 0.03 # m

current = output["Beam/current"] ##(1, n_s)

y = output["Beam/yposition"] ##(n_z, n_s)
x = output["Beam/xposition"] ##(n_z, n_s)

ysize = output["Beam/ysize"] ##(n_z, n_s)
xsize = output["Beam/xsize"] ##(n_z, n_s)

energy = output["Beam/energy"] ##(n_z, n_s)
ku = output["Lattice/ku"] 

# Some part of the beam has been calculated at all. We need to discard this meaningless part.
# if np.where(np.abs(y[0,:])<1e-20)[0].size ==0:
#     cutoff_1 = 0
#     cutoff_2 = y.shape[1]-1
# else:
#     # We need to be very careful here. Both head and tail may conatin some parts that haven't been calculated at all.
#     cutoff_id = np.argmax(np.diff(np.where(np.abs(y[0,:])<1e-20)[0]))
#     cutoff_1 = np.where(np.abs(y[0,:])<1e-20)[0][cutoff_id]+1 # The last zero element in the head
#     cutoff_2 = np.where(np.abs(y[0,:])<1e-20)[0][cutoff_id+1]-1 # The first zero element in the tail
threshold = 500 # The threshold current
cutoff_1 = np.where(current[0,:]>threshold)[0][0]
cutoff_2 = np.where(current[0,:]>threshold)[0][-1]


y = y[:, cutoff_1:cutoff_2]
x = x[:, cutoff_1:cutoff_2]
ysize = ysize[:, cutoff_1:cutoff_2]
xsize = xsize[:, cutoff_1:cutoff_2]
on_axis_id = np.argmin(np.power(x, 2)+np.power(y, 2), axis =1)

energy = energy[:, cutoff_1:cutoff_2]

# Indices of slices in z where all undulators begin
UND_id = np.where(np.diff(ku)>0)[0]+1 


# Use orbit correctors to find all three undulator sections
cx = output["Lattice/cx"] ## (n_z, 1)
OC_id = np.ravel(np.nonzero(cx)) ## Usually this list has 6 elements since we have 3 pairs of orbit correctors.

# We are interested in the enegry loss of the core part of the beam.
E_axis_U1 = np.mean(energy[OC_id[1]:OC_id[2], :], axis = 1)

K_U1 = np.sqrt(4*E_axis_U1**2*lambdaref/lambuda_u-2)
aw_U1 = K_U1/np.sqrt(2)

## Calculate the desired taper in all undulators
taper_U1 = aw_U1[UND_id[1:8]-OC_id[1]]

plt.figure()
plt.plot(E_axis_U1)
plt.ylabel('Average Eenrgy of Bunch/$\gamma$')
plt.title('First Undulator Section')
plt.show()

## Fit the taper

n_fit = 6
z = np.polyfit(np.arange(n_fit), taper_U1[0:n_fit], 3)
p = np.poly1d(z)

taper_U1_fit = p(np.arange(n_fit+1))

plt.figure()
plt.plot(taper_U1)
plt.plot(p(np.arange(n_fit+1)))
plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
plt.title('First Undulator Section')
plt.show()
