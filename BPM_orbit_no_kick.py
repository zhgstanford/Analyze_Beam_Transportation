import h5py
import numpy as np
import matplotlib.pyplot as plt
from LCLS_beamline import *

if __name__ == "__main__":

    N_bin = 500
    lambdaref = 1.84631767e-09 # Central XFEL wavelength

    ps_beg = np.zeros((4, len(bunch["t"])))

    ps_beg[0, :] = bunch["x"]
    ps_beg[1, :] = bunch["xp"]
    ps_beg[2, :] = bunch["y"]
    ps_beg[3, :] = bunch["yp"]

    id_slices, zplot, _ = flip_slice(bunch["t"], bins = N_bin)

    ##########
    beamline_id = Orbit_Corrector
    ps_end = analyze_phase_space_at_end(ps_beg, LCLS_U3, beamline_id, id_slices, N_bin)

    ds_slice = np.average(np.diff(zplot))
    analyze_on_axis(ps_end, 2, 8, ds_slice, zplot)
    analyze_on_axis(ps_end, 10, 15, ds_slice, zplot)
    analyze_on_axis(ps_end, 17, 23, ds_slice, zplot)

    # Here we plot the center-of-mass orbit at each BPM. In my code, I plot the center-of-mass orbit at
    # the end of each quadrupole.
    x_BPM = np.ravel(np.nanmean(ps_end[0, :, :], axis=1))*1e6 # m to um
    y_BPM = np.ravel(np.nanmean(ps_end[2, :, :], axis=1))*1e6 # m to um

    plt.subplot(2, 1, 1)
    plt.plot(x_BPM)
    plt.xlabel('Id of BPMS')
    plt.ylabel('Center-of-Mass Orbits in X')

    plt.subplot(2, 1, 2)
    plt.plot(y_BPM)
    plt.xlabel('Id of BPMS')
    plt.ylabel('Center-of-Mass Orbits in Y')
    plt.tight_layout()
    plt.savefig('BPM_Orbits.jpg')
    plt.show()
