import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import constants

bunch = h5py.File('/home/zhaohengguo/Desktop/GENESIS4_Input/LCLS2SXR_DECHin_no_OC.bun', 'r')
gamma = bunch['pCentral'][()] # The central energy of the bunch

class Undulator:
    def __init__(self, lambdau, nwig, aw):
        L = lambdau*nwig
        self.length = L
        K = np.sqrt(2)*aw
        self.K = K

        self.ku = 2*np.pi/lambdau

        K_f = (self.K*self.ku/(np.sqrt(2)*gamma))**2

        if K == 0:
            self.M1 = [[1, L, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, L],
                       [0, 0, 0, 1]]
        else:
            self.M1 = [[1, L, 0, 0],
                       [0, 1, 0, 0],
                       [0 , 0 , np.cos(np.sqrt(K_f)*L), 1/np.sqrt(K_f)*np.sin(np.sqrt(K_f)*L)],
                       [0 , 0 , -np.sqrt(K_f)*np.sin(np.sqrt(K_f)*L), np.cos(np.sqrt(K_f)*L)]]



        self.M2 = [[0],
                   [0],
                   [0],
                   [0]]

class Quadrupole:
    def __init__(self, L, K):
        self.length = L
        self.focusing = K
        if K > 0:
            self.M1 = [[np.cos(np.sqrt(K)*L), 1/np.sqrt(K)*np.sin(np.sqrt(K)*L), 0, 0],
                       [-np.sqrt(K)*np.sin(np.sqrt(K)*L), np.cos(np.sqrt(K)*L), 0, 0],
                       [0, 0, np.cosh(np.sqrt(K)*L), 1/np.sqrt(K)*np.sinh(np.sqrt(K)*L)],
                       [0, 0, np.sqrt(K)*np.sinh(np.sqrt(K)*L), np.cosh(np.sqrt(K)*L)]]
        else:
            K = np.abs(K)
            self.M1 = [[np.cosh(np.sqrt(K)*L), 1/np.sqrt(K)*np.sinh(np.sqrt(K)*L), 0, 0],
                       [np.sqrt(K)*np.sinh(np.sqrt(K)*L), np.cosh(np.sqrt(K)*L),   0, 0],
                       [0 , 0 , np.cos(np.sqrt(K)*L), 1/np.sqrt(K)*np.sin(np.sqrt(K)*L)],
                       [0 , 0 , -np.sqrt(K)*np.sin(np.sqrt(K)*L), np.cos(np.sqrt(K)*L)]]

        self.M2 = [[0],
                   [0],
                   [0],
                   [0]]

class Drift:
    def __init__(self, L):
        self.length = L
        self.M1 = [[1, L, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, L],
                   [0, 0, 0, 1]]
        self.M2 = [[0],
                   [0],
                   [0],
                   [0]]

class Orbit_Corrector:
    def __init__(self, L, cx, cy):
        self.length = L
        self.kick_x = cx
        self.lick_y = cy
        self.M1 = [[1, L, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, L],
                   [0, 0, 0, 1]]
        self.M2 = [[0.5*cx*L],
                   [cx],
                   [0.5*cy*L],
                   [cy]]

class Chicane:
    def __init__(self, L, lb, ld, delay):
        self.length = L
        self.length_dipole = lb
        self.length_drift_dipole = ld
        self.length_delay = delay

        theta = np.sqrt(3*delay/(2*lb))
        M_dipole = [[1, lb, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, np.cos(theta), (lb/theta)*np.sin(theta)],
                   [0, 0, -(theta/lb)*np.sin(theta), np.cos(theta)]]
        M_drift = [[1, L-4*lb, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, L-4*lb],
                   [0, 0, 0, 1]]

        self.M1 = np.linalg.multi_dot([M_dipole, M_dipole, M_drift, M_dipole, M_dipole])
        self.M2 = [[0],
                   [0],
                   [0],
                   [0]]

def beam_transportation(init_phase_spc, undulator_line):
    """Input:
    init_phase_space: A 4-by-N array, which is the 4D phase space of the bunch. N
    is the number of macroparticles in the bunch.
    undulator_line: A list which contains components defined in the previous section."""

    phase_spc = init_phase_spc

    for element in undulator_line:
        phase_spc = np.dot( element.M1, phase_spc ) +element.M2

    return phase_spc

def emittance_and_twiss(ps_4D):
    """Input:
    ps_4D: A 4-by-N array, which is the 4D phase space of the bunch. N
    is the number of macroparticles in the bunch.
    Output:
    alpha_x, beta_x, gamma_x, emittance_x, alpha_y, beta_y, gamma_y, emittance_y:
    All are real numbers.
    """
    x = ps_4D[0,:]
    xp = ps_4D[1,:]
    y = ps_4D[2,:]
    yp = ps_4D[3,:]

    if (len(x) <= 10):
        emittance_x = 0
        alpha_x = 0
        beta_x = 0
        gamma_x = 0

        emittance_y = 0
        alpha_y = 0
        beta_y = 0
        gamma_y = 0
    else:
        # Remove the average value before calculating its Twiss parameters.
        x = x-x.mean()
        xp = xp-xp.mean()
        y = y-y.mean()
        yp = yp-yp.mean()

        emittance_x = np.sqrt(np.average(x**2)*np.average(xp**2)-(np.average(np.multiply(x, xp)))**2)
        # alpha_x = -(np.average(np.multiply(x, xp))-np.average(x)*np.average(xp))/emittance_x
        # beta_x = (np.average(x**2)-np.average(x)**2)/emittance_x
        # gamma_x = (np.average(xp**2)-np.average(xp)**2)/emittance_x
        alpha_x = -np.cov(x, xp)[0, 1]/emittance_x
        beta_x = np.std(x)**2/emittance_x
        gamma_x = np.std(xp)**2/emittance_x

        emittance_y = np.sqrt(np.average(y**2)*np.average(yp**2)-(np.average(np.multiply(y, yp)))**2)
        # alpha_y = -(np.average(np.multiply(y, yp))-np.average(y)*np.average(yp))/emittance_y
        # beta_y = (np.average(y**2)-np.average(y)**2)/emittance_y
        # gamma_y = (np.average(yp**2)-np.average(yp)**2)/emittance_y
        alpha_y = -np.cov(y, yp)[0, 1]/emittance_y
        beta_y = np.std(y)**2/emittance_y
        gamma_y = np.std(yp)**2/emittance_y

    return emittance_x, alpha_x, beta_x, gamma_x, emittance_y, alpha_y, beta_y, gamma_y

def flip_slice(t, bins=2944):
    """We use this function to flip the bunch in s and convert the information from
    time (t) to the longitudinal length along the bunch(s).
    Input:
    t: An array which has the length N. N is the number of particles.
    Output:
    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.
    zplot: A array of length N_bin. It is the array of bin edges in s."""

    s = constants.c*np.ravel((np.max(t)-t)) # Flip s
    s_sort_index = np.argsort(s) # Ascending order.

    plt.clf()

    hist_min = np.min(s)
    # hist_max = np.min(s)+(np.max(s)-np.min(s))*0.96
    hist_max = np.min(s)+(np.max(s)-np.min(s))*0.99
    hist, zplot, patches = plt.hist(s, bins,range = (hist_min, hist_max))

    id_slices = []
    for n in range(0, bins):
        num_begin = int(np.sum(hist[0:n]))
        num_end = int(np.sum(hist[0:n+1]))
        id_slices.append(s_sort_index[num_begin:num_end])

    return id_slices, zplot, hist

def beam_property_along_s(ps, id_slices):
    """We use this function to analyze the beam property along s.
    Input:
    ps: A 4-by-N array. N is the number of macro-particles.
    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.
    Output:
    prop_s: A 14-by-N_bin array. For each column, elements are: average of x, px, y, py, x_RMS
    y_RMS, and emittance_x, alpha_x, beta_x, gamma_x, emittance_y, alpha_y, beta_y, gamma_y."""

    prop_s = np.zeros((14, len(id_slices)))
    for n in range(len(id_slices)):
        ps_s = np.take(ps, id_slices[n], axis=1)
        prop_s[0, n] = np.average(ps_s[0,:])
        prop_s[1, n] = np.average(ps_s[1,:])
        prop_s[2, n] = np.average(ps_s[2,:])
        prop_s[3, n] = np.average(ps_s[3,:])
        prop_s[4, n] = np.std(ps_s[0,:])
        prop_s[5, n] = np.std(ps_s[2,:])
        prop_s[6:, n] = emittance_and_twiss(ps_s)

    return prop_s


def analyze_phase_space_at_end(ps_beg, beamline, beamline_id, id_slices, N_bin):
    """I want to use this function to calculate the 4D phase space at the end of
    some components in the undulator beamline.

    Input:
    ps_beg: A 4-by-N array. N is the number of macro-particles.

    beamline: A list which contains components defined in the previous section.

    beamline_id: It might be Quadrupole, Chicane, Undulator or Orbit_Corrector.

    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.

    N_bin: It is the number of mesh grids in the histogram.

    Output:
    ps_end: A 4-by-N-by-N_bin array. N is the number of components labelled
    by 'beamline_id' in the beamline. N_bin is the number of mesh grids in s,
    the longitudinal coordinate of the bunch.
    """

    phase_spc = ps_beg

    ## Count how many times the class beamline_id occurs in the beamline_id
    count_beamline = 0
    for element in beamline:
        if isinstance(element, beamline_id):
                count_beamline += 1
    ps_end = np.zeros((4, count_beamline, N_bin))

    count_id = 0
    for element in beamline:
        #print(element)
        phase_spc = np.dot( element.M1, phase_spc ) +element.M2
        if isinstance(element, beamline_id):
            ps_along_s = beam_property_along_s(phase_spc, id_slices)
            ps_end[0, count_id, :] = ps_along_s[0, :]
            ps_end[1, count_id, :] = ps_along_s[1, :]
            ps_end[2, count_id, :] = ps_along_s[2, :]
            ps_end[3, count_id, :] = ps_along_s[3, :]
            count_id += 1

    return ps_end

def analyze_on_axis(phase_space, id_begin, id_end, ds_slice, zplot):
    """ We want to use this function to analyze which part is on the axis in
    each undulator section.

    Input:

    phase_space: A 4-by-N-by-M array, which N is the number of beamline elements
    and M is the number of mesh grids in s. We recommend to calculate the phase
    space distribution at the end of each quadrupole.

    id_begin and id_end: Two integers. The phase space distribution we want to
    analzye is phase_space[:, (id_begin-1):id_end, :].

    ds_slice: A positive number in unit of meter. It is the length of one
    longitudinal slice in s.

    Output:

    s_on_axis: A psotive number in unit of meter. It shows which part is on the
    axis in this undulator section."""

    ps = phase_space[:, (id_begin-1):id_end, :]
    # print(np.shape(ps))
    # ps = ps[numpy.logical_not(numpy.isnan(ps))]

    x = ps[0, :, :]
    px = ps[1, :, :]
    y = ps[2, :, :]
    py = ps[3, :, :]

    id_on_axis = np.zeros((4, int(id_end-id_begin+1)))

    for n in range(int(id_end-id_begin+1)):
        x_this = x[n, :]
        px_this = px[n, :]
        y_this = y[n, :]
        py_this = py[n, :]

        # Remove all NAN elements in the phase space array
        x_this = x_this[np.logical_not(np.isnan(x_this))]
        px_this = px_this[np.logical_not(np.isnan(px_this))]
        y_this = y_this[np.logical_not(np.isnan(y_this))]
        py_this = py_this[np.logical_not(np.isnan(py_this))]

        ## Plot X
        plt.subplot(2, 2, 1)
        plt.plot(zplot[0:len(x_this)]*1e+6, x_this*1e+6)
        plt.ylabel('Position in X/ $\mu$m', fontsize=10)

        ## Plot Y
        plt.subplot(2, 2, 2)
        plt.plot(zplot[0:len(y_this)]*1e+6, y_this*1e+6)
        plt.ylabel('Position in Y/ $\mu$m', fontsize=10)

        ## Plot px
        plt.subplot(2, 2, 3)
        plt.plot(zplot[0:len(px_this)]*1e+6, px_this)
        plt.ylabel('Angle in X', fontsize=10)

        ## Plot py
        plt.subplot(2, 2, 4)
        plt.plot(zplot[0:len(py_this)]*1e+6, py_this)
        plt.ylabel('Angle in Y', fontsize=10)


        # plt.xlabel('Longitudianl Direction of the Bunch $s$/ $\mu$m')
        # plt.title('First Undulator Section')
        # plt.title('Second Undulator Section')
        # plt.title('Third Undulator Section')

        id_on_axis[0, n] = np.argmin(np.abs(x_this))
        id_on_axis[1, n] = np.argmin(np.abs(px_this))
        id_on_axis[2, n] = np.argmin(np.abs(y_this))
        id_on_axis[3, n] = np.argmin(np.abs(py_this))

    fig = plt.gcf()
    fig.set_size_inches(13.5, 9)
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    fig.savefig('phase_space_U3_new.png', dpi=100)
    plt.show()


    s_on_axis = np.average(id_on_axis[2:4,:])*ds_slice

    return id_on_axis, s_on_axis

def analyze_orbit_corrector(OC1, OC2, beamline, phase_beg):
    """We want to use this function to optimize parameters for orbit correctors.
    Indeed, we always need two orbit correctors to do this job.

    Input:
    OC1 and OC2: Two orbit correctors. Please notice that they may have different
    lengths.

    beamline: A list of all beamline components between those two orbit correctors.
    Please notice that we define the travel direction of the bunch is from the left
    to the rightself.

    phase_beg: A 4-by-1 array. This is the 4D phase space position of the slice
    which we want to make on the axis at the end of the second orbit correctors.

    Output:
    OC_parameters: A 4-by-1 array, which is (cx_1, cx_2, cy_1, cy_2).
    """

    M = np.identity(4)
    OC_parameters = np.zeros(4)

    for element in beamline:
        M = np.dot(element.M1, M)

    # Since the X and Y are decoupled, we can treat them separately.
    M_x = M[0:2, 0:2]
    M_y = M[2:4, 2:4]

    L1 = [[OC1.length/2], [1]]
    L2 = [[OC2.length/2], [1]]

    M_OC1 = np.array(OC1.M1)[0:2, 0:2]
    M_OC2 = np.array(OC2.M1)[0:2, 0:2]

    # The following part solve the cx_1 and cx_2
    M1_x = np.linalg.multi_dot([M_OC2, M_x, L1])
    M2_x = np.linalg.multi_dot([M_OC2, M_x, M_OC1])
    M_OC_x = np.hstack((M1_x, L2))

    OC_parameters[0:2] = -np.linalg.multi_dot([np.linalg.inv(M_OC_x), M2_x, phase_beg[0:2]])
    # The end of the X-part

    # The following part solve the cy_1 and cy_2
    M1_y = np.linalg.multi_dot([M_OC2, M_y, L1])
    M2_y = np.linalg.multi_dot([M_OC2, M_y, M_OC1])
    M_OC_y = np.hstack((M1_y, L2))

    OC_parameters[2:4] = -np.linalg.multi_dot([np.linalg.inv(M_OC_y), M2_y, phase_beg[2:4]])
    # The end of the Y-part


    return OC_parameters

def set_up_orbit_correctors(ps_beg, delay, id_slice1, ds_slice, zplot, id_slices, U_core, lambdaref):
    """I want to use this function to directly set up all orbit correctors.

    Input:
    ps_beg: A 4-by-N array. N is the number of macro-particles.

    delay: A array with two elements. They are lengths of delays in two chicanes.
    Its unit is meter.

    id_slice1: A integer. It is the id of the slices that radiates in the first undulator section.

    ds_slices: The length of one slice in s. Unit: m.

    zplot: A 1-by-M array, where M is the number of mesh grids in z. It is used for making plots.

    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.

    U_core: A list of three beamlines. The core part of each undulator beamline doesn't
    contain any orbit correctors.

    lambdaref: The central frequency of the XFEL radiation. Unit: m.
    """
    SXSS = Chicane(3, 0.1, 0.0, delay[0])
    HXSS = Chicane(3, 0.1, 0.0, delay[1])

    OC1 = [CORR00, UND01, D1, QUAD01, CORR01]
    OC2 = [CORR08, SXSS, SXSS_PHASE, D2, QUAD09, CORR09]
    OC3 = [CORR15, HXSS, HXSS_PHASE, D1, QUAD16, CORR16]


    # Calculate the phase space distribution of each slice at the undulator entrance.
    ps_beg_slice1 = beam_property_along_s(ps_beg, id_slices)[0:4, :]

    ps_on_axis_1 = np.ravel(ps_beg_slice1[:, id_slice1])

    # Set up the first pair of orbit correctors.
    OC1_optimized = analyze_orbit_corrector(OC1[0], OC1[-1], OC1[1:-1], ps_on_axis_1)
    print(OC1_optimized)
    CORR00_new = Orbit_Corrector(OC1[0].length, OC1_optimized[0], OC1_optimized[2])
    CORR01_new = Orbit_Corrector(OC1[-1].length, OC1_optimized[1], OC1_optimized[3])

    # The whole U1 with optimized orbit correctors
    U1_new = [CORR00_new] + OC1[1:-1] + [CORR01_new] + U_core[0]
    ps_end1 = beam_transportation(ps_beg, U1_new)

    # ps_end1 is a 4-by-N array. N is the number of macro-particles. It is the full
    # 4D phase space distribution at the end of the first undulator section.

    # The id of the slice on the axis in the second undulator section
    on_axis_id_U2 = int(id_slice1+delay[0]/ds_slice+ (8*110)*lambdaref/ds_slice) # The last part is slippage

    print(on_axis_id_U2)

    ps_end_slice1 = beam_property_along_s(ps_end1, id_slices)[0:4, :]
    ps_on_axis_2 = np.ravel(ps_end_slice1[:, on_axis_id_U2])

    OC2_optimized = analyze_orbit_corrector(OC2[0], OC2[-1], OC2[1:-1], ps_on_axis_2)
    print(OC2_optimized)
    CORR08_new = Orbit_Corrector(OC2[0].length, OC2_optimized[0], OC2_optimized[2])
    CORR09_new = Orbit_Corrector(OC2[-1].length, OC2_optimized[1], OC2_optimized[3])

    # The whole U2 with optimized orbit correctors
    U2_new = [CORR08_new] + OC2[1:-1] + [CORR09_new] + U_core[1]
    ps_end2 = beam_transportation(ps_end1, U2_new)

    # ps_end2 is a 4-by-N array. N is the number of macro-particles. It is the full
    # 4D phase space distribution at the end of the second undulator section.

    # The id of the slice on the axis in the third undulator section
    on_axis_id_U3 = int(id_slice1+(delay[0]+delay[1])/ds_slice +(14*110*lambdaref)/ds_slice) # The last term is the slipage

    print(on_axis_id_U3)

    ps_end_slice2 = beam_property_along_s(ps_end2, id_slices)[0:4, :]
    ps_on_axis_3 = np.ravel(ps_end_slice2[ :, on_axis_id_U3])

    OC3_optimized = analyze_orbit_corrector(OC3[0], OC3[-1], OC3[1:-1], ps_on_axis_3)
    print(OC3_optimized)
    CORR15_new = Orbit_Corrector(OC3[0].length, OC3_optimized[0], OC3_optimized[2])
    CORR16_new = Orbit_Corrector(OC3[-1].length, OC3_optimized[1], OC3_optimized[3])

    U3_new = [CORR15_new] + OC3[1:-1] + [CORR16_new] + U_core[2]

    Undulator_Beamline = U1_new+U2_new+U3_new

    return Undulator_Beamline, OC2_optimized, OC3_optimized


## Initialize all the components

### QUAD ###
QUAD01=Quadrupole(0.078,-2.97177294204433)
QUAD02=Quadrupole(0.078,+2.97177294204433)
QUAD03=Quadrupole(0.078,-2.97177294204433)
QUAD04=Quadrupole(0.078,+2.97177294204433)
QUAD05=Quadrupole(0.078,-2.97177294204433)
QUAD06=Quadrupole(0.078,+2.97177294204433)
QUAD07=Quadrupole(0.078,-2.97177294204433)
QUAD08=Quadrupole(0.078,+2.97177294204433)
QUAD09=Quadrupole(0.078,-2.97177294204433)
QUAD10=Quadrupole(0.078,+2.97177294204433)
QUAD11=Quadrupole(0.078,-2.97177294204433)
QUAD12=Quadrupole(0.078,+2.97177294204433)
QUAD13=Quadrupole(0.078,-2.97177294204433)
QUAD14=Quadrupole(0.078,+2.97177294204433)
QUAD15=Quadrupole(0.078,-2.97177294204433)
QUAD16=Quadrupole(0.078,+2.97177294204433)
QUAD17=Quadrupole(0.078,-2.97177294204433)
QUAD18=Quadrupole(0.078,+2.97177294204433)
QUAD19=Quadrupole(0.078,-2.97177294204433)
QUAD20=Quadrupole(0.078,+2.97177294204433)
QUAD21=Quadrupole(0.078,-2.97177294204433)
QUAD22=Quadrupole(0.078,+2.97177294204433)
QUAD23=Quadrupole(0.078,-2.97177294204433)
QUAD24=Quadrupole(0.078,+2.97177294204433)

### Drift ###
D1 = Drift( 0.261 )
D2 = Drift( 0.411 )

### Undulator ###

UND00 = Undulator(0.03, 110, 2.4748)
UND01 = Undulator(0.03, 110, 2.4750)
UND02 = Undulator(0.03, 110, 2.4746)
UND03 = Undulator(0.03, 110, 2.4742)
UND04 = Undulator(0.03, 110, 2.4738)
UND05 = Undulator(0.03, 110, 2.4734)
UND06 = Undulator(0.03, 110, 2.4690)
UND07 = Undulator(0.03, 110, 2.4690)
UND08 = Undulator(0.03, 110, 2.4690)

UND10 = Undulator(0.03, 110, 2.47999 )
UND11 = Undulator(0.03, 110, 2.47998 )
UND12 = Undulator(0.03, 110, 2.47997 )
UND13 = Undulator(0.03, 110, 2.47996 )
UND14 = Undulator(0.03, 110, 2.47800 )
UND15 = Undulator(0.03, 110, 2.47800 )

UND17 = Undulator(0.03, 110, 2.49 )
UND18 = Undulator(0.03, 110, 2.4857 )
UND19 = Undulator(0.03, 110, 2.4808000000000003 )
UND20 = Undulator(0.03, 110, 2.4753000000000003 )
UND21 = Undulator(0.03, 110, 2.4692000000000003 )
UND22 = Undulator(0.03, 110, 2.4625000000000004 )
UND23 = Undulator(0.03, 110, 2.4552 )
UND24 = Undulator(0.03, 110, 2.4473000000000003 )
UND25 = Undulator(0.03, 110, 2.4613600000000004 )
UND26 = Undulator(0.03, 110, 2.45476 )
UND27 = Undulator(0.03, 110, 2.4471600000000002 )
UND28 = Undulator(0.03, 110, 2.43856 )
UND29 = Undulator(0.03, 110, 2.4289600000000005 )
UND30 = Undulator(0.03, 110, 2.41836 )
UND31 = Undulator(0.03, 110, 2.4067600000000002 )
UND32 = Undulator(0.03, 110, 0.0 )

### Orbit Corrector ###
# CORR00 = Orbit_Corrector( 0.001, 0e-6, -75e-6)
# CORR01 = Orbit_Corrector( 0.411, 0, 0)

CORR00 = Orbit_Corrector( 0.001, -4.20302333e-06, -6.22946775e-05)
CORR01 = Orbit_Corrector( 0.411, 5.67436428e-06, -4.76823760e-06)
CORR02 = Orbit_Corrector( 0.411, 0, 0)
CORR03 = Orbit_Corrector( 0.411, 0, 0)
CORR04 = Orbit_Corrector( 0.411, 0, 0)
CORR05 = Orbit_Corrector( 0.411, 0, 0)
CORR06 = Orbit_Corrector( 0.411, 0, 0)
CORR07 = Orbit_Corrector( 0.411, 0, 0)

# CORR08 = Orbit_Corrector( 0.411, -5e-6, -6e-5)
# CORR09 = Orbit_Corrector( 0.411, 0, 2.2e-5)
CORR08 = Orbit_Corrector( 0.411, -1.03675773e-05, -3.42800735e-05)
CORR09 = Orbit_Corrector( 0.411, 8.45248529e-06, 1.36110114e-05)
CORR10 = Orbit_Corrector( 0.411, 0, 0)
CORR11 = Orbit_Corrector( 0.411, 0, 0)
CORR12 = Orbit_Corrector( 0.411, 0, 0)
CORR13 = Orbit_Corrector( 0.411, 0, 0)
CORR14 = Orbit_Corrector( 0.411, 0, 0)

CORR15 = Orbit_Corrector( 0.411, -8.41412685e-06, 2.66302968e-05)
CORR16 = Orbit_Corrector( 0.411, 7.80166049e-06, 1.22830809e-05)

CORR17 = Orbit_Corrector( 0.411, 0, 0)
CORR18 = Orbit_Corrector( 0.411, 0, 0)
CORR19 = Orbit_Corrector( 0.411, 0, 0)
CORR20 = Orbit_Corrector( 0.411, 0, 0)
CORR21 = Orbit_Corrector( 0.411, 0, 0)
CORR22 = Orbit_Corrector( 0.411, 0, 0)
CORR23 = Orbit_Corrector( 0.411, 0, 0)
CORR24 = Orbit_Corrector( 0.411, 0, 0)

### Chicane ###
SXSS = Chicane( 3, 0.1, 0.0, 2.75e-06)

HXSS = Chicane( 3, 0.1, 0.0, 6e-06)

### PHASESHIFTER
SXSS_PHASE = Drift(0.3)
HXSS_PHASE = Drift(0.3)

### Useful Beamline ###

U1 = [ CORR00,
UND01, D1, QUAD01, CORR01,
UND02, D1, QUAD02, CORR02,
UND03, D2, QUAD03, CORR03,
UND04, D1, QUAD04, CORR04,
UND05, D1, QUAD05, CORR05,
UND06, D2, QUAD06, CORR06,
UND07, D1, QUAD07, CORR07,
UND08, D1, QUAD08, CORR08]


U3 = [ CORR00,
UND01, D1, QUAD01, CORR01,
UND02, D1, QUAD02, CORR02,
UND03, D2, QUAD03, CORR03,
UND04, D1, QUAD04, CORR04,
UND05, D1, QUAD05, CORR05,
UND06, D2, QUAD06, CORR06,
UND07, D1, QUAD07, CORR07,
UND08, D1, QUAD08, CORR08,
SXSS,  D2, QUAD09, CORR09,
UND10, D1, QUAD10, CORR10,
UND11, D1, QUAD11, CORR11,
UND12, D2, QUAD12, CORR12,
UND13, D1, QUAD13, CORR13,
UND14, D1, QUAD14, CORR14,
UND15, D2, QUAD15, CORR15,
HXSS,  D1, QUAD16, CORR16,
UND17, D1, QUAD17, CORR17,
UND18, D2, QUAD18, CORR18,
UND19, D1, QUAD19, CORR19,
UND20, D1, QUAD20, CORR20,
UND21, D2, QUAD21, CORR21,
UND22, D1, QUAD22, CORR22,
UND23, D1, QUAD23, CORR23,
UND24, D2, QUAD24, CORR24]

U3_test = [ CORR00,
UND01, D1, QUAD01, CORR01,
UND02, D1, QUAD02, CORR02,
UND03, D2, QUAD03, CORR03,
UND04, D1, QUAD04, CORR04,
UND05, D1, QUAD05, CORR05,
UND06, D2, QUAD06, CORR06,
UND07, D1, QUAD07, CORR07,
UND08, D1, QUAD08, CORR08,
SXSS, SXSS_PHASE,  D2, QUAD09, CORR09,
UND10, D1, QUAD10, CORR10,
UND11, D1, QUAD11, CORR11,
UND12, D2, QUAD12, CORR12,
UND13, D1, QUAD13, CORR13,
UND14, D1, QUAD14, CORR14,
UND15, D2, QUAD15, CORR15,
HXSS, HXSS_PHASE, D1, QUAD16, CORR16,
UND17, D1, QUAD17, CORR17,
UND18, D2, QUAD18, CORR18,
UND19, D1, QUAD19, CORR19,
UND20, D1, QUAD20, CORR20,
UND21, D2, QUAD21, CORR21,
UND22, D1, QUAD22, CORR22,
UND23, D1, QUAD23, CORR23,
UND24, D2, QUAD24, CORR24]


## Here I want to set up a beamline to simulate the by-center orbit when we just don't set up any orbit corrector at all.
## Unfortunately, since in Marc's code he uniformly shifted the beam, here we need to play with the first pair of orbit
## correctors to make sure that the head is on the axis.

CORR00_no_kick = Orbit_Corrector( 0.001, 0, 0)
CORR01_no_kick = Orbit_Corrector( 0.411, 0, 0)

CORR08_no_kick = Orbit_Corrector( 0.411, 0, 0)
CORR09_no_kick = Orbit_Corrector( 0.411, 0, 0)

CORR15_no_kick = Orbit_Corrector( 0.411, 0, 0)
CORR16_no_kick = Orbit_Corrector( 0.411, 0, 0)

U3_no_kick = [ CORR00_no_kick,
UND01, D1, QUAD01, CORR01_no_kick,
UND02, D1, QUAD02, CORR02,
UND03, D2, QUAD03, CORR03,
UND04, D1, QUAD04, CORR04,
UND05, D1, QUAD05, CORR05,
UND06, D2, QUAD06, CORR06,
UND07, D1, QUAD07, CORR07,
UND08, D1, QUAD08, CORR08_no_kick,
SXSS, SXSS_PHASE,  D2, QUAD09, CORR09_no_kick,
UND10, D1, QUAD10, CORR10,
UND11, D1, QUAD11, CORR11,
UND12, D2, QUAD12, CORR12,
UND13, D1, QUAD13, CORR13,
UND14, D1, QUAD14, CORR14,
UND15, D2, QUAD15, CORR15_no_kick,
HXSS, HXSS_PHASE, D1, QUAD16, CORR16_no_kick,
UND17, D1, QUAD17, CORR17,
UND18, D2, QUAD18, CORR18,
UND19, D1, QUAD19, CORR19,
UND20, D1, QUAD20, CORR20,
UND21, D2, QUAD21, CORR21,
UND22, D1, QUAD22, CORR22,
UND23, D1, QUAD23, CORR23,
UND24, D2, QUAD24, CORR24]

## Set up the beamline part by part. The core part of each undulator beamline doesn't
## contain any orbit correctors.

U1_core = [UND02, D1, QUAD02, CORR02,
UND03, D2, QUAD03, CORR03,
UND04, D1, QUAD04, CORR04,
UND05, D1, QUAD05, CORR05,
UND06, D2, QUAD06, CORR06,
UND07, D1, QUAD07, CORR07,
UND08, D1, QUAD08]

U2_core = [UND10, D1, QUAD10, CORR10,
UND11, D1, QUAD11, CORR11,
UND12, D2, QUAD12, CORR12,
UND13, D1, QUAD13, CORR13,
UND14, D1, QUAD14, CORR14,
UND15, D2, QUAD15]

U3_core = [UND17, D1, QUAD17, CORR17,
UND18, D2, QUAD18, CORR18,
UND19, D1, QUAD19, CORR19,
UND20, D1, QUAD20, CORR20,
UND21, D2, QUAD21, CORR21,
UND22, D1, QUAD22, CORR22,
UND23, D1, QUAD23, CORR23,
UND24, D2, QUAD24, CORR24]

U_core = [U1_core, U2_core, U3_core]




## Begin to write some file

# ps_beg = np.zeros((4, len(bunch["t"])))

# ps_beg[0, :] = bunch["x"]
# ps_beg[1, :] = bunch["xp"]
# ps_beg[2, :] = bunch["y"]
# ps_beg[3, :] = bunch["yp"]

# id_slices, zplot = flip_slice(bunch["t"], bins = 200)
