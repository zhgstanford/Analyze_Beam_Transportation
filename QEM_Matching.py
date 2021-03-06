import numpy as np
import matplotlib.pyplot as plt
from twiss_matrix_transportation import *
from LCLS_beamline import *
from scipy.optimize import minimize

def QEM_matrix(k1, k2, k3, k4):
    D1 = Drift(2.07246)
    D2 = Drift(4.14492)
    D3 = Drift(12.14492)
    QEM1 = Quadrupole(0.316,k1)
    QEM2 = Quadrupole(0.316,k2)
    QEM3 = Quadrupole(0.316,k3)
    QEM4 = Quadrupole(0.316,k4)

    Matching_Beamline = [D1, QEM1, D2, QEM2, 
            D3, QEM3, D2, QEM4, D1]

    M = np.identity(4)
    for element in Matching_Beamline:
        M = np.dot(element.M1, M)

    return M

def transportation_matrix(k1, k2, k3, k4):
    M = QEM_matrix(k1, k2, k3, k4)

    Matching_End = np.array([[1.251215137881939e+00, -3.351564258427984e+01, 0.0, 0.0],
            [-3.538979594916994e-02,  1.747190939412409e+00,  0.0, 0.0],
            [0.0,  0.0, -8.577933984345419e-01, -6.305922371634640e+00],
            [0.0,  0.0, -5.462243508716558e-02, -1.567329426716162e+00]])

    M_total = np.matmul(Matching_End, M)

    return M_total

def QEM_mismatching(K_QEM):

    k1, k2, k3, k4 = K_QEM
    M_total = transportation_matrix(k1, k2, k3, k4)

    # The Twiss at the start of the QEM beamline
    T_start = np.array([3.1674, 48.8946, 0.2256368,3.2144, 79.6244, 0.14232279]).reshape((6,1))
    
    # The Twiss without any matching at the UNDBEG. K1 to K4 are set to be default values in the ELEGANT beamline
    # T_end0 = np.array([1.1428, 34.8047,0.066255, -0.4157, 18.6314, 0.0629478]).reshape((6,1))
    
    # The Twiss of the machine at UNDBEG
    T_target = np.array([1.72219023, 14.767043, 0.268567, -0.527199, 4.4073, 0.2899573]).reshape((6,1))

    T_end1 = twiss_after_transportation(M_total, T_start)
    loss = BMAG(T_end1, T_target)

    #print(T_end1)
    return loss



if __name__ == '__main__':
    
    k1 = -3.948193191E-01
    k2 = 4.370293743E-01
    k3 = -6.012049020E-01
    k4 = 4.256096075E-01

    x0 = np.array([k1, k2, k3, k4])
    res = minimize(QEM_mismatching, x0)

    print(res.x)
