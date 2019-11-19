import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import constants

bunch = h5py.File('/home/zhaohengguo/Desktop/GENESIS4_Input/LCLS2SXR_DECHin_with_OC.bun', 'r')
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

def set_up_orbit_correctors(ps_beg, delay, id_slice1, ds_slice, zplot, id_slices, U_core):
    """I want to use this function to directly set up all orbit correctors.

    Input:
    ps_beg: A 4-by-N array. N is the number of macro-particles.

    delay: A real number in unit of um.

    id_slice1: A integer. It is the id of the slices that radiates in the first undulator section.

    ds_slices: The length of one slice in s. Unit: m.

    zplot: A 1-by-M array, where M is the number of mesh grids in z. It is used for making plots.

    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.

    U_core: A list of three beamlines. The core part of each undulator beamline doesn't
    contain any orbit correctors.

    """

    U1_core, U2_core = U_core

    Chicane = [CORR08, D3, D1, SXRSS, D2, QSXH35, CORR09]

    ps_end1 = beam_transportation(ps_beg, U1_core)

    # The id of the slice on the axis in the second undulator section
    on_axis_id_U2 = int(id_slice1+delay/ds_slice)

    print(on_axis_id_U2)
    ps_end_slice1 = beam_property_along_s(ps_end1, id_slices)[0:4, :]
    ps_on_axis_2 = np.ravel(ps_end_slice1[:, on_axis_id_U2])

    OC2_optimized = analyze_orbit_corrector(Chicane[0], Chicane[-1], Chicane[1:-1], ps_on_axis_2)
    print(OC2_optimized)

    CORR08_new = Orbit_Corrector(Chicane[0].length, OC2_optimized[0], OC2_optimized[2])
    CORR09_new = Orbit_Corrector(Chicane[-1].length, OC2_optimized[1], OC2_optimized[3])

    Undulator_Beamline = U1_core + [CORR08_new] + Chicane[1:-1] + [CORR09_new] + U2_core

    return Undulator_Beamline, OC2_optimized


## Initialize all the components

### QUAD ###
QSXH26 = Quadrupole(0.084, 1.784478917)
QSXH27 = Quadrupole(0.084, -1.784478917)
QSXH28 = Quadrupole(0.084, 1.784478917)
QSXH29 = Quadrupole(0.084, -1.784478917)
QSXH30 = Quadrupole(0.084, 1.784478917)
QSXH31 = Quadrupole(0.084, -1.784478917)
QSXH32 = Quadrupole(0.084, 1.784478917)
QSXH33 = Quadrupole(0.084, -1.784478917)
QSXH34 = Quadrupole(0.084, 1.784478917)
QSXH35 = Quadrupole(0.084, -1.784478917)
QSXH36 = Quadrupole(0.084, 1.784478917)
QSXH37 = Quadrupole(0.084, -1.784478917)
QSXH38 = Quadrupole(0.084, 1.784478917)
QSXH39 = Quadrupole(0.084, -1.784478917)
QSXH40 = Quadrupole(0.084, 1.784478917)
QSXH41 = Quadrupole(0.084, -1.784478917)
QSXH42 = Quadrupole(0.084, 1.784478917)
QSXH43 = Quadrupole(0.084, -1.784478917)
QSXH44 = Quadrupole(0.084, 1.784478917)
QSXH45 = Quadrupole(0.084, -1.784478917)
QSXH46 = Quadrupole(0.084, 1.784478917)
QSXH47 = Quadrupole(0.084, -1.784478917)

### Drift ###
D1 = Drift( 0.1335 )
D2 = Drift( 0.4615 )
D3 = Drift( 0.328 )

### Undulator ###

UND00 = Undulator(0.03, 110, 2.4748)


UMASXH26 = Undulator(0.039, 87, 2.616540022441109)
UMASXH27 = Undulator(0.039, 87, 2.616540022441109)
UMASXH28 = Undulator(0.039, 87, 2.616540022441109)
UMASXH29 = Undulator(0.039, 87, 2.616540022441109)
UMASXH30 = Undulator(0.039, 87, 2.616540022441109)
UMASXH31 = Undulator(0.039, 87, 2.616540022441109)
UMASXH32 = Undulator(0.039, 87, 2.616540022441109)
UMASXH33 = Undulator(0.039, 87, 2.616540022441109)
UMASXH34 = Undulator(0.039, 87, 2.616540022441109)

UMASXH36 = Undulator(0.039, 87, 2.616540022441109)
UMASXH37 = Undulator(0.039, 87, 2.616540022441109)
UMASXH38 = Undulator(0.039, 87, 2.616540022441109)
UMASXH39 = Undulator(0.039, 87, 2.616540022441109)
UMASXH40 = Undulator(0.039, 87, 2.616540022441109)
UMASXH41 = Undulator(0.039, 87, 2.616540022441109)
UMASXH42 = Undulator(0.039, 87, 2.616540022441109)
UMASXH43 = Undulator(0.039, 87, 2.616540022441109)
UMASXH44 = Undulator(0.039, 87, 2.616540022441109)
UMASXH45 = Undulator(0.039, 87, 2.616540022441109)
UMASXH46 = Undulator(0.039, 87, 2.616540022441109)
UMASXH47 = Undulator(0.039, 87, 2.616540022441109)

# Orbit Correctors

CORR00 = Orbit_Corrector( 0.001, 0, 0)
CORR01 = Orbit_Corrector( 0.001, 0, 0)
CORR02 = Orbit_Corrector( 0.001, 0, 0)
CORR03 = Orbit_Corrector( 0.001, 0, 0)
CORR04 = Orbit_Corrector( 0.001, 0, 0)
CORR05 = Orbit_Corrector( 0.001, 0, 0)
CORR06 = Orbit_Corrector( 0.001, 0, 0)
CORR07 = Orbit_Corrector( 0.001, 0, 0)

CORR08 = Orbit_Corrector( 0.001, 0, 0)
CORR09 = Orbit_Corrector( 0.001, 0, 0)
CORR10 = Orbit_Corrector( 0.001, 0, 0)
CORR11 = Orbit_Corrector( 0.001, 0, 0)
CORR12 = Orbit_Corrector( 0.001, 0, 0)
CORR13 = Orbit_Corrector( 0.001, 0, 0)
CORR14 = Orbit_Corrector( 0.001, 0, 0)

CORR15 = Orbit_Corrector( 0.001, 0, 0)
CORR16 = Orbit_Corrector( 0.001, 0, 0)

CORR17 = Orbit_Corrector( 0.001, 0, 0)
CORR18 = Orbit_Corrector( 0.001, 0, 0)
CORR19 = Orbit_Corrector( 0.001, 0, 0)
CORR20 = Orbit_Corrector( 0.001, 0, 0)
CORR21 = Orbit_Corrector( 0.001, 0, 0)
CORR22 = Orbit_Corrector( 0.001, 0, 0)
CORR23 = Orbit_Corrector( 0.001, 0, 0)
CORR24 = Orbit_Corrector( 0.001, 0, 0)

### Chicane ###
SXRSS = Chicane( 3.4, 0.1, 0.0, 0.01e-06)


### Useful Beamline ###
LCLS2_SXR = [D1, UMASXH26, D2, QSXH26, CORR00, D3,
D1, UMASXH27, D2, QSXH27, CORR01, D3,
D1, UMASXH28, D2, QSXH28, CORR02, D3,
D1, UMASXH29, D2, QSXH29, CORR03, D3,
D1, UMASXH30, D2, QSXH30, CORR04, D3,
D1, UMASXH31, D2, QSXH31, CORR05, D3,
D1, UMASXH32, D2, QSXH32, CORR06, D3,
D1, UMASXH33, D2, QSXH33, CORR07, D3,
D1, UMASXH34, D2, QSXH34, CORR08, D3,
D1, SXRSS   , D2, QSXH35, CORR09, D3,
D1, UMASXH36, D2, QSXH36, CORR10, D3,
D1, UMASXH37, D2, QSXH37, CORR11, D3,
D1, UMASXH38, D2, QSXH38, CORR12, D3,
D1, UMASXH39, D2, QSXH39, CORR13, D3,
D1, UMASXH40, D2, QSXH40, CORR14, D3,
D1, UMASXH41, D2, QSXH41, CORR15, D3,
D1, UMASXH42, D2, QSXH42, CORR16, D3,
D1, UMASXH43, D2, QSXH43, CORR17, D3,
D1, UMASXH44, D2, QSXH44, CORR18, D3,
D1, UMASXH45, D2, QSXH45, CORR19, D3,
D1, UMASXH46, D2, QSXH46, CORR20, D3,
D1, UMASXH47, D2, QSXH47, CORR21, D3]


## Set up the beamline part by part. The core part of each undulator beamline doesn't
## contain any orbit correctors.

U1_core = [D1, UMASXH26, D2, QSXH26, CORR00, D3,
D1, UMASXH27, D2, QSXH27, CORR01, D3,
D1, UMASXH28, D2, QSXH28, CORR02, D3,
D1, UMASXH29, D2, QSXH29, CORR03, D3,
D1, UMASXH30, D2, QSXH30, CORR04, D3,
D1, UMASXH31, D2, QSXH31, CORR05, D3,
D1, UMASXH32, D2, QSXH32, CORR06, D3,
D1, UMASXH33, D2, QSXH33, CORR07, D3,
D1, UMASXH34, D2, QSXH34]

U2_core = [D3,
D1, UMASXH36, D2, QSXH36, CORR10, D3,
D1, UMASXH37, D2, QSXH37, CORR11, D3,
D1, UMASXH38, D2, QSXH38, CORR12, D3,
D1, UMASXH39, D2, QSXH39, CORR13, D3,
D1, UMASXH40, D2, QSXH40, CORR14, D3,
D1, UMASXH41, D2, QSXH41, CORR15, D3,
D1, UMASXH42, D2, QSXH42, CORR16, D3,
D1, UMASXH43, D2, QSXH43, CORR17, D3,
D1, UMASXH44, D2, QSXH44, CORR18, D3,
D1, UMASXH45, D2, QSXH45, CORR19, D3,
D1, UMASXH46, D2, QSXH46, CORR20, D3,
D1, UMASXH47, D2, QSXH47, CORR21, D3]

Chicane = [CORR08, D3,
D1, SXRSS   , D2, QSXH35, CORR09]

U_core = [U1_core, U2_core]
