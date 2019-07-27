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



    id_slices, zplot, hist = flip_slice(bunch["t"], bins = N_bin)

    id_slice1 = 80
    delay = np.array([1.2e-06, 1.20e-06])

    c = 299792458  # Speed of the light
    bunch_length = c*np.ptp(bunch["t"])
    ds_len = bunch_length/N_bin # The length of one slice in s
    Undulator_Beamline = set_up_orbit_correctors(ps_beg, delay, id_slice1, ds_len, zplot, id_slices, U_core, lambdaref)

    # We also want to get the phase space distribution along the bunch at the undulator entrance.
    ps_beg_s = beam_property_along_s(ps_beg, id_slices)

    ##########
    beamline_id = Quadrupole
    ps_end = analyze_phase_space_at_end(ps_beg, Undulator_Beamline, beamline_id, id_slices, N_bin)

    ds_slice = np.average(np.diff(zplot))
    analyze_on_axis(ps_end, 2, 8, ds_slice, zplot)
    analyze_on_axis(ps_end, 10, 15, ds_slice, zplot)
    analyze_on_axis(ps_end, 17, 24, ds_slice, zplot)

    # Here we plot the center-of-mass orbit at each BPM. In my code, I plot the center-of-mass orbit at
    # the end of each quadrupole.

    x_whole_BPM_mine = ps_end[0, :, :]
    y_whole_BPM_mine = ps_end[2, :, :]

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
