import numpy as np 
import matplotlib.pyplot as plt 
import h5py

output = h5py.File('/home/zhaohengguo/Desktop/Analyze_Beam_Transportation/GENESIS4_Output/scan_taper_U2_22.out.h5', 'r')

lambdaref = output["Global/lambdaref"][()] # m
lambda_u = 0.03 # m

y = output["Beam/yposition"] ##(n_z, n_s)
x = output["Beam/xposition"] ##(n_z, n_s)

ysize = output["Beam/ysize"] ##(n_z, n_s)
xsize = output["Beam/xsize"] ##(n_z, n_s)

energy = output["Beam/energy"] ##(n_z, n_s)
ku = output["Lattice/ku"] 

# Some part of the beam has been calculated at all. We need to discard this meaningless part.
if np.where(np.abs(y[0,:])<1e-20)[0].size ==0:
    cutoff = y.shape[1]-1
else:
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

K_U1 = np.sqrt(4*E_axis_U1**2*lambdaref/lambda_u-2)
aw_U1 = K_U1/np.sqrt(2)

K_U2 = np.sqrt(4*E_axis_U2**2*lambdaref/lambda_u-2)
aw_U2 = K_U2/np.sqrt(2)

## Calculate the desired taper in all undulators
taper_U1 = aw_U1[UND_id[1:8]-OC_id[1]]
taper_U2 = aw_U2[UND_id[8:14]-OC_id[3]]

plt.figure()
plt.plot(E_axis_U2)
plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
plt.title('Second Undulator Section')
plt.show()

## However, we may need to take the slippage into consideration. At different undulator, the location 
## of the peak power might be different.

# taper_U2_slippage = np.zeros(6)

# for n in range(6):
#     E_peak = energy[UND_id[8+n], U2_on_axis_id+n*11]
#     K_peak = np.sqrt(4*E_peak**2*lambdaref/lambda_u-2)
#     aw_peak = K_peak/np.sqrt(2)
#     taper_U2_slippage[n] = aw_peak

# print(taper_U2_slippage)


## Fit the taper

# n_fit = 5
# z = np.polyfit(np.arange(n_fit), taper_U2[0:n_fit], 3)
# p = np.poly1d(z)

# plt.figure()
# plt.plot(taper_U2)
# plt.plot(p(np.arange(n_fit+1)))
# plt.ylabel('Eenrgy of the on-axis slice/$\gamma$')
# plt.title('Second Undulator Section')
# plt.show()
