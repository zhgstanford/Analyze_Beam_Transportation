import numpy as np
import scipy.constants as const
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks

def select_freq_window(N, ds_len, E_width):
        """
        This function outputs the first and the last indices of the frequency in the frequency window.
        
        Input:
        N: A positive integer, which is the number of mesh grids in E_field.
        ds_len: An positive number with unit meter. It is the length of one mesh grid in E_field.
        E_width: The width of the spectrum window in eV.

        Output:
        spec_id: An 1-by-2 array, which is [id_start, id_end].
        """

        dE = const.h*const.c/(N*ds_len*const.e) # Unit eV. It is the length of one mesh grid in the spectrum.
        N_width = np.round(E_width/(2.*dE))*2 # In order to make sure that this is an even number.

        spec_id = np.array([N//2-N_width/2, N//2+N_width/2])
        print("Energy resolution is "+str(dE)+" eV.")

        return np.ravel(spec_id.astype(int))

def calculate_spectrum(E_field, ds_len, E_c):
        """
        We use this function to calculate the full spectrum of the farfield electric field.
        
        Input:
        E_field: An 1-by-N array of complex numbers, where N is the number of mesh grids in the time domain.
        ds_len: An positive number with unit meter. It is the length of one mesh grid in E_field.
        E_c: An positive number with unit eV. It is the central energy of the XFEL raidiation.
        
        Output:
        spectrum: An 1-by_N array of complex numbers.
        spectrum_axis: An 1-by_N array. It shows the corresponding photon energy for the spectrum. 
        """

        spectrum = np.roll(np.fft.fft(E_field), len(E_field)//2)
        freq = np.fft.fftfreq(len(E_field))
        spectrum_axis = np.roll(freq*const.h*const.c/(ds_len*const.e), len(E_field)//2)+E_c # tranform from frequency to eV

        return spectrum, spectrum_axis


def Fun_MGauss(VLEN, PARM):
    """ Input:
        VLEN: A positive integer, which is the length of the spectrum array.
        PARM: A array of length 3N, where N is the number of Gaussian in the fitting. For each Gaussian
              with the for a*np.exp(-(x-b)**2/(2*c**2)), the data structure is [a, b, c]. 
        Output:
        OUT: A array of length len(VLEN). It is the combination of all Gaussians."""

    OUT = 0
    for II in range(len(PARM)//3):
        OUT=OUT+PARM[3*II]*np.exp(-(np.arange(VLEN) - PARM[3*II+1])**2/2/(PARM[3*II+2]**2))
    return OUT

def MGauss_Err(PARM, TARGET):
    """ Input:
        TARGET: A array of length N. By default, this is the ground truth of the spectrum.
        PARM: A array of length 3N, where N is the number of Gaussian in the fitting. For each Gaussian
              with the for a*np.exp(-(x-b)**2/(2*c**2)), the data structure is [a, b, c]. 
        Output:
        Out: A positive number. It is the loss between the true spectrum and our fitting spectrum generated
        by the multi-Gaussian fitting."""
    OUT=(TARGET - Fun_MGauss(len(TARGET),PARM))
    OUT=np.sum(OUT**2)

    # Add penalties when any parameters become negative
    l1 = 1
    OUT += l1*np.linalg.norm(np.minimum(0, PARM), ord=2)

    return OUT

def Fit_MGauss(TARGET, PARM0, METHOD):
    OUT = minimize(MGauss_Err, x0 = PARM0, args = (TARGET), method = METHOD, options={'gtol': 1e-6, 'disp': True})
    FIT = Fun_MGauss(len(TARGET),OUT.x)
    return OUT, FIT

def SpikeCounter(SPEC, METHOD, THRES):
    # LP_GAUSS = np.exp(-np.arange(-20,20)**2/(2*5**2))
    # LP_GAUSS = LP_GAUSS/np.sum(LP_GAUSS)

    # Low pass filter
    #SPECLP = np.convolve(SPEC,LP_GAUSS)
    SPECLP = SPEC
    DIFFSPEC = np.diff(SPECLP)

    NODE = np.where(np.multiply(DIFFSPEC[0:-2], DIFFSPEC[1:-1])<0)[0]+1

    PEAK = np.where(np.logical_and(np.multiply(DIFFSPEC[0:-2], DIFFSPEC[1:-1])<0, DIFFSPEC[0:-2]>0))[0]+1
    VALLEY = np.where(np.logical_and(np.multiply(DIFFSPEC[0:-2], DIFFSPEC[1:-1])<0, DIFFSPEC[0:-2]<0))[0]+1
    
    MAX = np.max(SPECLP)
    THRESHOLD = MAX/THRES
    
    REALSPIKES=np.where(SPECLP[PEAK]>THRESHOLD)[0]
    LOC=PEAK[REALSPIKES]
    AMP=SPECLP[LOC]

    SIGMA=np.ones(len(LOC))*np.mean(np.diff(LOC))/5

    try:
        del PARM
    except Exception:
        pass

    PARM0 = np.zeros(3*len(LOC))
    PARM0[2::3] = SIGMA
    PARM0[0:-1:3] = AMP
    PARM0[1:-1:3] = LOC

    [FitParameters, FitFunction] = Fit_MGauss(SPECLP, PARM0, METHOD)

    return SPECLP, FitParameters, FitFunction

def count_peak(SPEC, PARM, threshold):
    """
    Input:
    SPEC: A array of length N. This is the spectrum that we applied the fitting to.
    PARM: A array of length 3M, where M is the number of Gaussian. For each Gaussian
          with the for a*np.exp(-(x-b)**2/(2*c**2)), the data structure is [a, b, c] and the
          integral of this Gaussian is np.sqrt(2*pi)*a*c.
    threshold: A positive number between 0 and 1. The number of spikes is counted as the minimum number 
               of Gaussian is required that the intergal of their sum is larger than threshold*np.sum(SPEC).
    Output:
    N_spikes: A positive integer.
    """

    area = np.sum(SPEC)
    
    Gaussian_a = PARM[0:-1:3]
    Gaussian_c = PARM[2::3]
    Gaussian_integral = np.ravel(np.multiply(Gaussian_a, Gaussian_c)*np.sqrt(2*np.pi))
    Gaussian_integral_sorted = np.sort(np.ravel(np.multiply(Gaussian_a, Gaussian_c)*np.sqrt(2*np.pi)))[::-1]
    MG_Sum = np.cumsum(Gaussian_integral_sorted)
    
    print(MG_Sum)
    
    print(np.where(MG_Sum>threshold*area))
    try:
        N_spikes = np.where(MG_Sum>threshold*area)[0][0]+1
    except Exception:
        N_spikes = len(PARM)//3

    if N_spikes == 1:
        PARM_reshape = PARM.reshape((-1, 3))
        PARM_rec = PARM_reshape[np.argsort(Gaussian_integral)[-1], :]
        PARM_new = np.ravel(PARM_rec)
        fitting_curve = Fun_MGauss(len(SPEC), PARM_new)
        peaks = int(np.ceil(PARM_new[1]))
    else:
        PARM_reshape = PARM.reshape((-1, 3))
        PARM_rec = PARM_reshape[np.argsort(Gaussian_integral)[-N_spikes:], :]
        PARM_new = np.ravel(PARM_rec)
        fitting_curve = Fun_MGauss(len(SPEC), PARM_new)

        peaks, _ = find_peaks(fitting_curve, prominence=0.05)

        N_spikes = peaks.shape[0]

    return N_spikes, fitting_curve, peaks


if __name__ == "__main__":
    output = h5py.File('/home/zhaohengguo/Desktop/test15/run4/scan_taper_U3_15.out.h5', 'r')
    E_c = const.h*const.c/(const.e*output["Global/lambdaref"][()])  # Central energy in eV
    intensity = output["Field/intensity-farfield"]
    phase = output["Field/phase-farfield"]
    zplot = output["Lattice/zplot"]
    ds_len = output["Global/lambdaref"][()]*10
    s_axis = np.arange(len(intensity[-1,:]))*ds_len

    E_field = np.sqrt(intensity[-1,:])*np.exp(1j*phase[-1,:])

    spectrum, spectrum_axis = calculate_spectrum(E_field, ds_len, E_c)

    spec_norm = np.max(np.abs(spectrum)**2)

    N = len(intensity[-1,:])
    E_width = 20 # eV
    spec_id = select_freq_window(N, ds_len, E_width)

    SPEC = np.abs(spectrum[spec_id[0]:spec_id[1]])**2/spec_norm
    
    SPECLP, FitParameters, FitFunction = SpikeCounter(SPEC, 'Nelder-Mead', 750)
    
    N_spikes, fitting_curve, peaks = count_peak(SPEC, FitParameters.x, 0.90)
    print(N_spikes)

    plt.plot(spectrum_axis[spec_id[0]:spec_id[1]], SPECLP, label='Spectrum from GENESIS 4')
    plt.plot(spectrum_axis[spec_id[0]:spec_id[1]], Fun_MGauss(len(SPECLP), FitParameters.x), label = 'Reconstructed Multi_Gaussian Spectrum')
    plt.plot(spectrum_axis[spec_id[0]:spec_id[1]], fitting_curve, label = 'Prominence Analysis')
    plt.plot((spectrum_axis[spec_id[0]:spec_id[1]])[peaks], fitting_curve[peaks], 'X', markersize= 10)
    plt.legend()
    plt.show()

    # plt.plot(spectrum_axis[spec_id[0]:spec_id[1]], np.abs(spectrum[spec_id[0]:spec_id[1]])**2/spec_norm)
    # plt.xlabel("Photon Energy eV")
    # plt.ylabel('Arbitrary Unit')
    # plt.title("XFEL Spectrum")
    # plt.show()
