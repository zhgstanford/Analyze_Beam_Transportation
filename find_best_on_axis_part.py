import h5py
import numpy as np
import matplotlib.pyplot as plt
from init_transportation import *

if __name__ == "__main__":
    
    c = 299792458  # Speed of the light

    bunch_length = c*np.ptp(bunch["t"])
    bunch_charge = bunch['Charge'][()]

    N_bin = 200
    ds_slen = bunch_length/N_bin # The length of one slice in s
    lambdaref = 1.84631767e-09 # Central XFEL wavelength for 670 eV

    ps_beg = np.zeros((4, len(bunch["t"])))

    ps_beg[0, :] = bunch["x"]
    ps_beg[1, :] = bunch["xp"]
    ps_beg[2, :] = bunch["y"]
    ps_beg[3, :] = bunch["yp"]

    id_slices, zplot, hist = flip_slice(bunch["t"], bins = N_bin)

    bunch_current = bunch_charge*hist/np.sum(hist)/(bunch_length/N_bin/c)
    
    ## ps_beg is a 4-by-N array which contains the phase space distribution of the input bunch file.
    ## For the ELEGANT output file and the GENESIS input file, the bunch head is on the left-hand-side.

    # Since we have already flip the t (or z), in the output plot the bunch head is on the right-hand-side.
    prop_s = beam_property_along_s(ps_beg, id_slices)

    alpha_x = prop_s[-7,:]
    beta_x = prop_s[-6,:]
    gamma_x = prop_s[-5,:]
    alpha_y = prop_s[-3,:]
    beta_y = prop_s[-2,:]
    gamma_y = prop_s[-1,:]

    plt.clf()

    # plt.plot(beta_x)
    # plt.show()
    # plt.title("Beta X along the bunch (head at RHS)")

    # plt.plot(beta_y)
    # plt.show()
    # plt.title("Beta Y along the bunch (head at RHS)")

    # Quick way to check which part is best matched to the machine
    # The following are Twiss parameters of the machine
    
    alpha_x0 = 1.3027986076894629
    beta_x0 = 17.178734895700195
    gamma_x0 = (1+alpha_x0**2)/beta_x0

    alpha_y0 = -0.6473999943278278
    beta_y0 = 8.163076828694416
    gamma_y0 = (1+alpha_y0**2)/beta_y0


    loss = (alpha_x-alpha_x0)**2+(beta_x-beta_x0)**2+(alpha_y-alpha_y0)**2+(beta_y-beta_y0)**2

    # Calculate BMAG in both X and Y to check mismatching effects
    BMAG_x = 0.5*(beta_x0*gamma_x-2*alpha_x0*alpha_x+gamma_x0*beta_x)
    BMAG_y = 0.5*(beta_y0*gamma_y-2*alpha_y0*alpha_y+gamma_y0*beta_y)

    threshold = 4

    good_slice=[i for i, x in enumerate(loss<threshold) if x]

    slice_U1 = 120
    slice_U2 = 240
    slice_U3 = 380

    slippage_1 = lambdaref*110*8
    slippage_2 = lambdaref*110*6

    # We need to subtract slipages from the total shift
    delay_1 = (slice_U2-slice_U1)*ds_slen-slippage_1
    delay_2 = (slice_U3-slice_U2)*ds_slen-slippage_2

    print('The delay in the first chicane is:')
    print(delay_1)
    print('The delay in the second chicane is:')
    print(delay_2)
