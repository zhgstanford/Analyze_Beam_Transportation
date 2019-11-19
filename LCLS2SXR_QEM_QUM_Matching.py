import numpy as np
import matplotlib.pyplot as plt
from twiss_matrix_transportation import *
from init_transportation import *
from scipy.optimize import minimize

def QEM_matrix(k1, k2, k3, k4):
    D1 = Drift(8.408634999999999)
    D2 = Drift(4.35412)
    D3 = Drift(12.35400000000027)
    D4 = Drift(4.35412)
    D5 = Drift(2.17799)
    QEM1 = Quadrupole(0.0535,k1)
    QEM2 = Quadrupole(0.0535,k2)
    QEM3 = Quadrupole(0.0535,k3)
    QEM4 = Quadrupole(0.0535,k4)

    Matching_Beamline = [D1, QEM1, QEM1, D2, QEM2, QEM2,
            D3, QEM3, QEM3, D4, QEM4, QEM4, D5]

    M = np.identity(4)
    for element in Matching_Beamline:
        M = np.dot(element.M1, M)

    return M

def QUM_matrix(k1, k2, k3, k4):
    D1 = Drift(7.08046000000013)
    D2 = Drift(5.157520000000204)
    D3 = Drift(8.407520000000204)
    D4 = Drift(4.407520000000204)
    D5 = Drift(9.977061999999933)

    QEM1 = Quadrupole(0.0534,k1)
    QEM2 = Quadrupole(0.0534,k2)
    QEM3 = Quadrupole(0.0534,k3)
    QEM4 = Quadrupole(0.0534,k4)

    Matching_Beamline = [D1, QEM1, QEM1, D2, QEM2, QEM2,
            D3, QEM3, QEM3, D4, QEM4, QEM4, D5]

    M = np.identity(4)
    for element in Matching_Beamline:
        M = np.dot(element.M1, M)

    return M


def QEM_mismatching(K_QEM):

    k1, k2, k3, k4 = K_QEM
    M_QEM = QEM_matrix(k1, k2, k3, k4)

    Twiss_QEMBEG = get_Twiss(8.8945, 32.7411, -0.2819, 4.7702)
    T_target = get_Twiss(20, 20, -0.9214, 1.1030)

    T_end = twiss_after_transportation(M_QEM, Twiss_QEMBEG)
    loss = BMAG(T_end, T_target)

    return loss


def get_Twiss(betax, betay, alphax, alphay):
    Twiss = np.zeros((6,1))
    Twiss[0,0] = alphax
    Twiss[1,0] = betax
    Twiss[2,0] = (1+alphax**2)/betax
    Twiss[3,0] = alphay
    Twiss[4,0] = betay
    Twiss[5,0] = (1+alphay**2)/betay

    return Twiss

def QUM_Match_Back(K_QUM):
    k1_QUM, k2_QUM, k3_QUM, k4_QUM = K_QUM

    [k1_QEM, k2_QEM, k3_QEM, k4_QEM] = [ 1.89128643, -1.91179331,  1.37507906, -1.10336075 ]
    M_QEM = QEM_matrix(k1_QEM, k2_QEM, k3_QEM, k4_QEM)
    M_QUM = QUM_matrix(k1_QUM, k2_QUM, k3_QUM, k4_QUM)
    M_QUM_END_to_SXRSTART = np.array([[-6.536135824554541e-01,  2.271736092325629e+01,  0, 0],
            [-6.693736829743772e-02,  7.965568171180256e-01,  0, 0],
            [0, 0, 7.121334880834014e-01,  2.345802226567454e+01],
            [0, 0, -9.123528744996316e-02, -1.601103477783533e+00]])

    M_DECHEND_to_QUM_BEG = np.array([[-1.487357282314065e+00, 4.510901413623954e+01,  0, 0],
            [-3.329149110341835e-02, 3.373408318006569e-01, 0, 0],
            [0, 0, 4.651046037459901e-02, 3.170933420184367e+01],
            [0, 0,-3.329149110341831e-02, -1.196526910888007e+00]])

    Twiss_QEMBEG = get_Twiss(8.8945, 32.7411, -0.2819, 4.7702)
    Twiss_SXRSTART = get_Twiss(12.9872, 12.9483, -0.6858, 1.1372)

    M_DECHEND_to_SXRSTART = M_QUM_END_to_SXRSTART.dot(M_QUM.dot(M_DECHEND_to_QUM_BEG))
    M_DECH = Drift(8.335876).M1
    M_total = np.dot(M_DECHEND_to_SXRSTART, np.dot(M_DECH, M_QEM))

    T_end1 = twiss_after_transportation(M_total, Twiss_QEMBEG)
    loss = BMAG(T_end1, Twiss_SXRSTART)

    return loss

def Dechirper_Objective(K_QEM_and_QUM):
    k1_QEM, k2_QEM, k3_QEM, k4_QEM = K_QEM_and_QUM[0:4]
    k1_QUM, k2_QUM, k3_QUM, k4_QUM = K_QEM_and_QUM[4:8]
    M_QEM = QEM_matrix(k1_QEM, k2_QEM, k3_QEM, k4_QEM)
    M_QUM = QUM_matrix(k1_QUM, k2_QUM, k3_QUM, k4_QUM)

    Twiss_QEMBEG = get_Twiss(8.8945, 32.7411, -0.2819, 4.7702)

    M_QEM1 = np.dot(Drift(2).M1, M_QEM)
    M_QEM2 = np.dot(Drift(4).M1, M_QEM)
    M_QEM3 = np.dot(Drift(6).M1, M_QEM)

    Twiss_DECHBEG = twiss_after_transportation(M_QEM, Twiss_QEMBEG)
    Twiss_DECHBEG1 = twiss_after_transportation(M_QEM1, Twiss_QEMBEG)
    Twiss_DECHBEG2 = twiss_after_transportation(M_QEM2, Twiss_QEMBEG)
    Twiss_DECHBEG3 = twiss_after_transportation(M_QEM3, Twiss_QEMBEG)

    loss1_0 = Twiss_DECHBEG[1]**2+Twiss_DECHBEG[4]**2
    loss1_1 = Twiss_DECHBEG1[1]**2+Twiss_DECHBEG1[4]**2
    loss1_2 = Twiss_DECHBEG2[1]**2+Twiss_DECHBEG2[4]**2
    loss1_3 = Twiss_DECHBEG3[1]**2+Twiss_DECHBEG3[4]**2


    M_QUM_END_to_SXRSTART = np.array([[-6.536135824554541e-01,  2.271736092325629e+01,  0, 0],
            [-6.693736829743772e-02,  7.965568171180256e-01,  0, 0],
            [0, 0, 7.121334880834014e-01,  2.345802226567454e+01],
            [0, 0, -9.123528744996316e-02, -1.601103477783533e+00]])

    M_DECHEND_to_QUM_BEG = np.array([[-1.487357282314065e+00, 4.510901413623954e+01,  0, 0],
            [-3.329149110341835e-02, 3.373408318006569e-01, 0, 0],
            [0, 0, 4.651046037459901e-02, 3.170933420184367e+01],
            [0, 0,-3.329149110341831e-02, -1.196526910888007e+00]])

    Twiss_SXRSTART = get_Twiss(12.9872, 12.9483, -0.6858, 1.1372)

    M_DECHEND_to_SXRSTART = M_QUM_END_to_SXRSTART.dot(M_QUM.dot(M_DECHEND_to_QUM_BEG))
    M_DECH = Drift(8.335876).M1
    M_total = np.dot(M_DECHEND_to_SXRSTART, np.dot(M_DECH, M_QEM))

    T_end1 = twiss_after_transportation(M_total, Twiss_QEMBEG)
    loss2 = BMAG(T_end1, Twiss_SXRSTART)

    loss = (loss2-0.88)*(loss1_0+loss1_1+loss1_2+loss1_3)

    print(Twiss_DECHBEG[1],Twiss_DECHBEG[4])
    print(Twiss_DECHBEG1[1],Twiss_DECHBEG1[4])
    print(Twiss_DECHBEG2[1],Twiss_DECHBEG2[4])
    print(Twiss_DECHBEG3[1],Twiss_DECHBEG3[4])
    print(loss2)

    return loss

def QEM_Core_Matching(K_QEM):
    # We want to optimize QEM to match the core part of the bunch to the machine
    k1_QEM, k2_QEM, k3_QEM, k4_QEM = K_QEM
    M_QEM = QEM_matrix(k1_QEM, k2_QEM, k3_QEM, k4_QEM)

    # We don;t change QUMs here
    k1_QUM = 3.488255192E-01
    k2_QUM =-1.300000000E-01
    k3_QUM = 3.836664830E-01
    k4_QUM =-6.298099012E-01
    M_QUM = QUM_matrix(k1_QUM, k2_QUM, k3_QUM, k4_QUM)

    M_QUM_END_to_SXRSTART = np.array([[-6.536135824554541e-01,  2.271736092325629e+01,  0, 0],
            [-6.693736829743772e-02,  7.965568171180256e-01,  0, 0],
            [0, 0, 7.121334880834014e-01,  2.345802226567454e+01],
            [0, 0, -9.123528744996316e-02, -1.601103477783533e+00]])

    M_DECHEND_to_QUM_BEG = np.array([[-1.487357282314065e+00, 4.510901413623954e+01,  0, 0],
            [-3.329149110341835e-02, 3.373408318006569e-01, 0, 0],
            [0, 0, 4.651046037459901e-02, 3.170933420184367e+01],
            [0, 0,-3.329149110341831e-02, -1.196526910888007e+00]])

    Twiss_QEMBEG = get_Twiss(13.7002, 20.4931, 0.2489, 3.0532)
    Twiss_SXRSTART = get_Twiss(12.9872, 12.9483, -0.6858, 1.1372)

    M_DECHEND_to_SXRSTART = M_QUM_END_to_SXRSTART.dot(M_QUM.dot(M_DECHEND_to_QUM_BEG))
    M_DECH = Drift(8.335876).M1
    M_total = np.dot(M_DECHEND_to_SXRSTART, np.dot(M_DECH, M_QEM))

    T_end1 = twiss_after_transportation(M_total, Twiss_QEMBEG)
    loss = BMAG(T_end1, Twiss_SXRSTART)

    return loss



if __name__ == '__main__':
    
    k1_QEM = 1.999441277E+00
    k2_QEM =-1.810331661E+00
    k3_QEM = 1.272436815E+00
    k4_QEM = -8.930980624E-01

    res = minimize(QEM_Core_Matching, [k1_QEM, k2_QEM, k3_QEM, k4_QEM])
    print(res.x)
    print(QEM_Core_Matching(res.x))