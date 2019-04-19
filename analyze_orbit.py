import h5py 
import numpy as np 
import matplotlib.pyplot as plt 

output = h5py.File("test3.out.h5", 'r')

n_s = len(np.ravel(output["Beam/current"]))

id_quad_end = np.ravel(np.nonzero(output["Lattice/qf"]))+1

phase_space_after_quad = np.zeros((4, len(id_quad_end), n_s))
phase_space_after_quad[0, :, :] = output["Beam/xposition"][id_quad_end, :]
phase_space_after_quad[1, :, :] = output["Beam/pxposition"][id_quad_end, :]
phase_space_after_quad[2, :, :] = output["Beam/yposition"][id_quad_end, :]
phase_space_after_quad[3, :, :] = output["Beam/pyposition"][id_quad_end, :]

# The following codes read the phase space distribution at the end of each undulator
id_und_end = np.ravel(np.where(np.diff(output["Lattice/aw"])<0))+1

phase_space_after_und = np.zeros((4, len(id_und_end), n_s))
phase_space_after_und[0, :, :] = output["Beam/xposition"][id_und_end, :]
phase_space_after_und[1, :, :] = output["Beam/pxposition"][id_und_end, :]
phase_space_after_und[2, :, :] = output["Beam/yposition"][id_und_end, :]
phase_space_after_und[3, :, :] = output["Beam/pyposition"][id_und_end, :]

# The following codes read the phase space distribution at the end of each
# nonzero orbit corrector.

id_oc_end = np.ravel(np.nonzero(output["Lattice/cy"]))+1

phase_space_after_oc = np.zeros((4, len(id_oc_end), n_s))
phase_space_after_oc[0, :, :] = output["Beam/xposition"][id_oc_end, :]
phase_space_after_oc[1, :, :] = output["Beam/pxposition"][id_oc_end, :]
phase_space_after_oc[2, :, :] = output["Beam/yposition"][id_oc_end, :]
phase_space_after_oc[3, :, :] = output["Beam/pyposition"][id_oc_end, :]

# The following codes read the phase space distribution at the end of each
# chicane

# id_chicane_end = np.ravel(np.nonzero(output["Lattice/chic_angle"]))+1
# phase_space_after_chicane = np.zeros((4, len(id_chicane_end), n_s))
# phase_space_after_chicane[0, :, :] = output["Beam/xposition"][id_chicane_end, :]
# phase_space_after_chicane[1, :, :] = output["Beam/pxposition"][id_chicane_end, :]
# phase_space_after_chicane[2, :, :] = output["Beam/yposition"][id_chicane_end, :]
# phase_space_after_chicane[3, :, :] = output["Beam/pyposition"][id_chicane_end, :]