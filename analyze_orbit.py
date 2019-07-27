import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from LCLS_beamline import *

output = h5py.File("/home/zhaohengguo/Desktop/GENESIS4_Output/UNDBEG_X13Y14.out.h5", 'r')
gamma = output['Global/gamma0'][()] 

n_s = len(np.ravel(output["Beam/current"]))

print(np.ravel(np.nonzero(output["Lattice/qf"])))
id_quad_end = np.ravel(np.nonzero(output["Lattice/qf"]))+1
id_quad_start = id_quad_end-1

phase_space_after_quad = np.zeros((4, len(id_quad_end), n_s))
phase_space_after_quad[0, :, :] = output["Beam/xposition"][id_quad_end, :]
phase_space_after_quad[1, :, :] = output["Beam/pxposition"][id_quad_end, :]/gamma
phase_space_after_quad[2, :, :] = output["Beam/yposition"][id_quad_end, :]
phase_space_after_quad[3, :, :] = output["Beam/pyposition"][id_quad_end, :]/gamma

phase_space_before_quad = np.zeros((4, len(id_quad_start), n_s))
phase_space_before_quad[0, :, :] = output["Beam/xposition"][id_quad_start, :]
phase_space_before_quad[1, :, :] = output["Beam/pxposition"][id_quad_start, :]/gamma
phase_space_before_quad[2, :, :] = output["Beam/yposition"][id_quad_start, :]
phase_space_before_quad[3, :, :] = output["Beam/pyposition"][id_quad_start, :]/gamma

df_M = QUAD01.M1
f_M = QUAD02.M1

slice_id = 333

for i in range(7):
    ps_start = phase_space_before_quad[:, i, slice_id]
    ps_end = phase_space_after_quad[:, i, slice_id]

    if i%2 == 0:
        print(i)
        print(ps_start)
        #print(np.matmul(df_M, np.matmul(df_M, ps_start)))
        print(ps_end)
    else:
        print(i)
        print(ps_start)
        #print(np.matmul(f_M, np.matmul(f_M, ps_start)))
        print(ps_end)

for i in range(6):
    x_start = phase_space_after_quad[0, i, slice_id]
    x_end = phase_space_before_quad[0, i+1, slice_id]
    xp = phase_space_after_quad[1, i, slice_id]

    L_x = (x_end-x_start)/xp

    print(i)
    print(x_start)
    print(x_end)
    print(xp)
    print(L_x)
    #print(L_y)

    
