from init_transportation import *

def plot_y_at_QUADs(ps_beg, beamline, gamma, id_slices, zplot):
    """ We want to use this function to anlyze the phase space distribution,
    especially py, before and after each undulator. Since the kick is mainly in Y,
    we need to take the natural focusing effect in Y direction into our
    consideration. In this fucntion, we calculate the bunch's transportation
    in the undulator beamline. We compare the py-s distribution before and after
    each undulator.

    Input:

    ps_beg: A 4-by-N array, which is the 4D phase space of the bunch. N
    is the number of macroparticles in the bunch.

    undulator_line: A list which contains components defined in the previous section.

    gamma: A positive number, which is the average gamma of the bunch.
    (The relativistic parameter, not the twiss parameter.)

    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.

    zplot: A array of length N_bin. It is the array of bin edges in s.
    """

    ps_before = ps_beg

    count_QUAD = 0

    for element in beamline:
        ps_after = np.dot( element.M1, ps_before ) +element.M2

        # Check whether this element is an undulatorself.
        if isinstance(element, Quadrupole):
            count_QUAD += 1
            # The phase space distribution along the bunch before and after the
            # bunch.
            ps_s_after = beam_property_along_s(ps_after, id_slices)

            save_name = 'Time_Dependent_Y_after_QUAD'+str(count_QUAD)
            plt.figure()
            plt.plot(zplot[0:-1], ps_s_after[2,:], label = 'After QUAD '+str(count_QUAD))
            plt.grid()
            plt.legend()
            plt.savefig(save_name)
            ## End if

        ps_before = ps_after

    return

if __name__ == "__main__":
    gamma = bunch["pCentral"][()]

    beamline = U1

    plot_y_at_QUADs(ps_beg, beamline, gamma, id_slices, zplot)
