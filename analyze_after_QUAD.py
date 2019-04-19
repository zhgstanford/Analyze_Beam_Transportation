import numpy as np
import matplotlib.pyplot as plt

id_quad = 7
gamma = 7835.445782948515


# phase_space_after_quad = np.load('phase_space_after_quad.npy')
# ps_end = np.load('ps_end.npy')

################

# py_GENESIS = phase_space_after_quad[3, id_quad, :]
# py_mine = ps_end[3, id_quad, :]*gamma
# norm_quad_py = np.max(np.abs(py_GENESIS))
#
# plt.plot(py_GENESIS, label = 'GENESIS 4 Result')
# plt.plot(py_mine, label = 'My Result')
# plt.ylim(-1,+1)
# # plt.legend()
# plt.figure()
# plt.plot((py_GENESIS-py_mine)[100:-1]/norm_quad_py)


y_GENESIS = phase_space_after_quad[2, id_quad, :]
y_mine = ps_end[2, id_quad, :]

norm_quad_y = np.max(np.abs(y_GENESIS))

plt.plot(y_GENESIS, label = 'GENESIS 4 Result')
plt.plot(y_mine, label = 'My Result')
plt.ylim(-1e-3, +1e-3)
# plt.legend()
plt.figure()
plt.plot((y_GENESIS-y_mine)[100:-1]/1e-3)

# px_GENESIS = phase_space_after_quad[1, id_quad, :]
# px_mine = ps_end[1, id_quad, :]*gamma
#
# norm_quad_px = np.max(np.abs(px_GENESIS))
#
# plt.plot(px_GENESIS, label = 'GENESIS 4 Result')
# plt.plot(px_mine, label = 'My Result')
# # plt.legend()
# plt.figure()
# plt.plot((px_GENESIS-px_mine)[100:-1]/norm_quad_px)

# x_GENESIS = phase_space_after_quad[0, id_quad, :]
# x_mine = ps_end[0, id_quad, :]
#
# norm_quad_x = np.max(np.abs(x_GENESIS))
#
# plt.plot(x_GENESIS, label = 'GENESIS 4 Result')
# plt.plot(x_mine, label = 'My Result')
# # plt.legend()
# plt.figure()
# plt.plot((x_GENESIS-x_mine)[100:-1]/norm_quad_x)
