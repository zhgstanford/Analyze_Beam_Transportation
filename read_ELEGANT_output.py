import numpy as np 
import matplotlib.pyplot as plt 
import re
import init_transportation as trans
import os

class phase_space:
    def __init__(self, fid, bins):
        with open(fid) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        self.bins = bins
        self.filename = fid
        self.Step = content[0] 
        self.pCentral = content[1] 
        self.Charge = content[2] 
        self.IDSlotsPerBunch = content[4]
        self.SVNVersion = content[5]
        self.Particles = int(content[6]) # Number of particles in the output file

        self.ps = np.zeros((7, self.Particles))

        for n in range(self.Particles):
            self.ps[:, n] = [float(i) for i in content[7+n].split()[0:7]]
            # x, xp, y, yp, t, gamma, particleID
        self.ps[4,:] -= self.ps[4,:].mean()

        self.id_slices,_ = trans.flip_slice(self.ps[4,:], self.bins)

    def remove_mean_value(self):
        # t = self.ps[4,:]
        # id_slices, _ = trans.flip_slice(t)

        ps_mean_removed = np.zeros(np.shape(self.ps))

        for n in range(len(self.id_slices)):
            if len(self.id_slices[n])>1:
                ps_s = np.take(self.ps, self.id_slices[n], axis=1)
                ps_mean_removed[0, self.id_slices[n]] = ps_s[0,:] -ps_s[0,:].mean()
                ps_mean_removed[1, self.id_slices[n]] = ps_s[1,:] -ps_s[1,:].mean()
                ps_mean_removed[2, self.id_slices[n]] = ps_s[2,:] -ps_s[2,:].mean()
                ps_mean_removed[3, self.id_slices[n]] = ps_s[3,:] -ps_s[3,:].mean()
                ps_mean_removed[4:6, self.id_slices[n]] = ps_s[4:6,:]
                # ps_mean_removed[5, id_slices[n]] = ps_s[5,:]

        return ps_mean_removed

    def plot_phase_space_slice(self, id_s, label, id_slices = None):
        # id_s is the index of the slice that we want to check
        if id_slices == None:
            t = self.ps[4,:]
            id_slices, _ = trans.flip_slice(t, self.bins) # Usually we set self.bins=100.
            ps_s = np.take(self.ps, id_slices[id_s], axis=1)
            print(len(id_slices))
            print('Id of slices along the bunch calculated.')
        else:
            print(len(id_slices))
            ps_s = np.take(self.ps, id_slices[id_s], axis=1)

        x  = np.ravel(ps_s[0,:] - ps_s[0,:].mean())*1e6 # um to m
        px = np.ravel(ps_s[1,:] - ps_s[1,:].mean())
        y  = np.ravel(ps_s[2,:] - ps_s[2,:].mean())*1e6 # um to m
        py = np.ravel(ps_s[3,:] - ps_s[3,:].mean())

        if label == 'x_2D':
            plt.close('all')
            plt.hist2d(x, px, bins=50, range = [[-100, +100], [-1e-5, +1e-5]])
            plt.xlabel('Position in X/ $\mu$m')
            plt.ylabel('Angle in X')
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_powerlimits((0,2))
            plt.title(self.filename)
            plot_name = self.filename+'_x_2D.png'
            plt.savefig(plot_name)
            # plt.show()
        elif label == 'y_2D':
            plt.close('all')
            plt.hist2d(y, py, bins=50, range = [[-100, +100], [-1e-5, +1e-5]])
            plt.xlabel('Position in Y/ $\mu$m')
            plt.ylabel('Angle in Y')
            ax = plt.gca()
            ax.yaxis.get_major_formatter().set_powerlimits((0,2))
            plt.title(self.filename)
            plot_name = self.filename+'_y_2D.png'
            plt.savefig(plot_name)
            #plt.show()
        elif label == 'x_1D':
            plt.close('all')
            label_x = 'std x:'+str(round(x.std(), 2))+'um'
            plt.hist(x, bins = 50 ,label = label_x)
            plt.xlabel('Position in X/ $\mu$m')
            plt.ylabel('Counts')
            plt.legend()
            plt.title(self.filename)
            plot_name = self.filename+'_x_1D.png'
            plt.savefig(plot_name)
            # plt.show()
        elif label == 'y_1D':
            plt.close('all')
            label_y = 'std y:'+str(round(y.std(), 2))+'um'
            plt.hist(y, bins = 50 ,label = label_y)
            plt.xlabel('Position in Y/ $\mu$m')
            plt.ylabel('Counts')
            plt.legend()
            plt.title(self.filename)
            plot_name = self.filename+'_y_1D.png'
            plt.savefig(plot_name)
            # plt.show()
        elif label == 'px_1D':
            plt.close('all')
            label_px = 'std px:'+'{:.3e}'.format(px.std())
            plt.hist(px, bins = 50 ,label = label_px)
            plt.xlabel('Angle in X')
            plt.ylabel('Counts')
            plt.legend()
            plt.title(self.filename)
            ax = plt.gca()
            ax.xaxis.get_major_formatter().set_powerlimits((0,2))
            plot_name = self.filename+'_px_1D.png'
            plt.savefig(plot_name)
            # plt.show()
        elif label == 'py_1D':
            plt.close('all')
            label_py = 'std py:'+'{:.3e}'.format(py.std())
            plt.hist(py, bins = 50 ,label = label_py)
            plt.xlabel('Angle in Y')
            plt.ylabel('Counts')
            plt.legend()
            plt.title(self.filename)
            ax = plt.gca()
            ax.xaxis.get_major_formatter().set_powerlimits((0,2))
            plot_name = self.filename+'_py_1D.png'
            plt.savefig(plot_name)
            # plt.show()



if __name__ == "__main__":

    data_path = '/Users/guozhaoheng/Desktop/Genesis4/GENESIS_output_file/530eV/Dechirper/'
    os.chdir(data_path)

    slice_id = 50
    ps_beg1 = phase_space('DECH1BEG.output', bins=100)
    ps_beg1.plot_phase_space_slice(slice_id, 'x_2D')
    ps_beg1.plot_phase_space_slice(slice_id, 'y_2D')
    ps_beg1.plot_phase_space_slice(slice_id, 'x_1D')
    ps_beg1.plot_phase_space_slice(slice_id, 'y_1D')
    ps_beg1.plot_phase_space_slice(slice_id, 'px_1D')
    ps_beg1.plot_phase_space_slice(slice_id, 'py_1D')

    ps_end1 = phase_space('DECH1END.output', bins=100)
    ps_end1.plot_phase_space_slice(slice_id, 'x_2D')
    ps_end1.plot_phase_space_slice(slice_id, 'y_2D')
    ps_end1.plot_phase_space_slice(slice_id, 'x_1D')
    ps_end1.plot_phase_space_slice(slice_id, 'y_1D')
    ps_end1.plot_phase_space_slice(slice_id, 'px_1D')
    ps_end1.plot_phase_space_slice(slice_id, 'py_1D')

    ps_beg2 = phase_space('DECH2BEG.output', bins=100)
    ps_beg2.plot_phase_space_slice(slice_id, 'x_2D')
    ps_beg2.plot_phase_space_slice(slice_id, 'y_2D')
    ps_beg2.plot_phase_space_slice(slice_id, 'x_1D')
    ps_beg2.plot_phase_space_slice(slice_id, 'y_1D')
    ps_beg2.plot_phase_space_slice(slice_id, 'px_1D')
    ps_beg2.plot_phase_space_slice(slice_id, 'py_1D')


    ps_end2 = phase_space('DECH2END.output', bins=100)
    ps_end2.plot_phase_space_slice(slice_id, 'x_2D')
    ps_end2.plot_phase_space_slice(slice_id, 'y_2D')
    ps_end2.plot_phase_space_slice(slice_id, 'x_1D')
    ps_end2.plot_phase_space_slice(slice_id, 'y_1D')
    ps_end2.plot_phase_space_slice(slice_id, 'px_1D')
    ps_end2.plot_phase_space_slice(slice_id, 'py_1D')
