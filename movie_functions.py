from init_transportation import *
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


################################################################################
################################################################################
##################################   Movie    ##################################
################################################################################
################################################################################
def make_movie(ps_beg, undulator_line, id_slices, zplot, plot_index, video_name):
    """The function makes a movie of the data along the undulator line (z).
    Input:
    ps_beg: An 4-by-N array, where N is the number of particles. This array is the
    4D phase space of the bunch.

    undulator_line: A list of beamline components.

    id_slices: A list of length N_bin, where N_bin is the number of mesh grids in
    the histogram. The list id_slices[i-1] contains the indices of partciles in the
    i-th mesh grid in s.

    zplot: A array of length N_bin. It is the array of bin edges in s. We need
    this to make a physical plot.

    plot_index: A real number. We need to specify which property to plot here.
    0 for x, 1 for px, 2 for y and 3 for py. We should first focus on these four
    quantites.

    video_name: A string which is the name of the output video.

    Output:
    A video that has the name video_name.

    This function replies on the function beam_property_along_s()."""

    xlab = 'Longitudinal Coordinate s/'+'$\mu$ m'

    if plot_index == 0:
        ylab = 'Transverse Coordinate X/ m'
        y_axis = 2.0e-4
    elif plot_index ==1:
        ylab = 'Angle xp'
        y_axis = 2.5e-5
    elif plot_index == 2:
        ylab = 'Transverse Coordinate Y/ m'
        y_axis = 0.0e-4
    elif plot_index == 3:
        ylab = 'Angle yp'
        y_axis = 1.0e-4

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1000)

    # fig = plt.figure()
    fig= plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')
    ims = []

    ps_beg_slice = beam_property_along_s(ps_beg, id_slices)
    data = ps_beg_slice[plot_index, :]

    # The following two variables define where the text is located
    x_axis = 0.8*np.max(zplot)*1e+6 # from m to um
    legend_str = 'UND BEG'

    print(x_axis)

    frame, = plt.plot(zplot[0:-1]*1e6, data*7835.445782948515, linewidth=2, markersize=12)
    # ims.append([frame, plt.title(legend_str)])
    ims.append([frame,
    plt.text(x_axis, y_axis, legend_str, bbox={'facecolor': 'white', 'pad': 6})])

    ims.append([frame])

    # plt.xlabel(xlab)
    # plt.ylabel(ylab)

    phase_spc = ps_beg

    count_QUAD = 0 ## Count the number of passed quadrupoles
    count_UND = 0 ## Count the number of passed undulaotrs
    for element in undulator_line:
        phase_spc = np.dot( element.M1, phase_spc ) +element.M2

        if isinstance(element, Quadrupole):
            count_QUAD += 1
            # If so, we want to check the beam property.
            ps_slice = beam_property_along_s(phase_spc, id_slices)
            data = ps_slice[plot_index, :]

            legend_str = 'After QUAD'+str(count_QUAD)

            frame, = plt.plot(zplot[0:-1]*1e6, data*7835.445782948515, linewidth=2, markersize=12, label = legend_str)

            ims.append([frame,
            plt.text(x_axis, y_axis, legend_str, bbox={'facecolor': 'white', 'pad': 6})])

        # if isinstance(element, Undulator):
        #     count_UND += 1
        #     # If so, we want to check the beam property.
        #     ps_slice = beam_property_along_s(phase_spc, id_slices)
        #     data = ps_slice[plot_index, :]
        #
        #     legend_str = 'After UND'+str(count_UND)
        #
        #     frame, = plt.plot(zplot[0:-1]*1e6, data, linewidth=2, markersize=12, label = legend_str)
        #
        #     ims.append([frame,
        #     plt.text(x_axis, y_axis, legend_str, bbox={'facecolor': 'white', 'pad': 6})])

    plt.ylim( -1.5e-4*7835.445782948515, 1.5e-4*7835.445782948515),
    plt.grid()
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    im_ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=3000,
                                   blit=True)
    im_ani.save(video_name, writer=writer, dpi = 100)

    return phase_spc


################################################################################
################################################################################
##################################   Movie    ##################################
################################################################################
################################################################################
