import numpy as np

def twiss_after_transportation(M, old_Twiss):
    """We use this function to calculate the new twiss after a beam transportation.
    Input:
    M: A 4-by-4 matrix. It contains 2 2-by-2 small blocks. All other components are zero,
    which means that this is no coupling between the X- and Y- direction.
    old_Twiss: A 6-by-1 array, whose elements are alpha_x, beta_x, gamma_x, alpha_y,
    beta_y, gamma_y.
    Output:
    new_Twiss:A 6-by-1 array, whose elements are alpha_x, beta_x, gamma_x, alpha_y,
    beta_y, gamma_y.
    """

    Mx = M[0:2, 0:2]
    My = M[2:4, 2:4]
    old_Twiss_x = old_Twiss[0:3,0]
    old_Twiss_y = old_Twiss[3:6,0]

    Mx_twiss = np.array([[Mx[0,0]*Mx[1,1]+Mx[0,1]*Mx[1,0], -Mx[0,0]*Mx[1,0], -Mx[0,1]*Mx[1,1]],
                        [-2*Mx[0,0]*Mx[0,1], Mx[0,0]**2, Mx[0,1]**2],
                        [-2*Mx[1,0]*Mx[1,1], Mx[1,0]**2, Mx[1,1]**2]])

    My_twiss = np.array([[My[0,0]*My[1,1]+My[0,1]*My[1,0], -My[0,0]*My[1,0], -My[0,1]*My[1,1]],
                        [-2*My[0,0]*My[0,1], My[0,0]**2, My[0,1]**2],
                        [-2*My[1,0]*My[1,1], My[1,0]**2, My[1,1]**2]])

    new_Twiss = np.zeros((6,1))

    new_Twiss[0:3, 0] = np.matmul(Mx_twiss, old_Twiss_x)
    new_Twiss[3:6, 0] = np.matmul(My_twiss, old_Twiss_y)

    return new_Twiss

def BMAG(Twiss1, Twiss2):
    alphax_1 = Twiss1[0,0]
    betax_1  = Twiss1[1,0]
    gammax_1 = Twiss1[2,0]
    alphay_1 = Twiss1[3,0]
    betay_1  = Twiss1[4,0]
    gammay_1 = Twiss1[5,0]

    alphax_2 = Twiss2[0,0]
    betax_2  = Twiss2[1,0]
    gammax_2 = Twiss2[2,0]
    alphay_2 = Twiss2[3,0]
    betay_2  = Twiss2[4,0]
    gammay_2 = Twiss2[5,0]

    BMAG_x = 0.5*(betax_1*gammax_2+gammax_1*betax_2-2*alphax_1*alphax_2)
    BMAG_y = 0.5*(betay_1*gammay_2+gammay_1*betay_2-2*alphay_1*alphay_2)

    BMAG = 0.5*(BMAG_x+BMAG_y)

    return BMAG



if __name__ == '__main__':
    Matching_Begin = np.array([[2.461993626855328e+00, 4.042237756015998e+01,0.0,0.0],
            [-1.187345243389780e-01, -1.543274414202793e+00,0.0,0.0],
            [0.0, 0.0, 6.543809971117196e-01, 4.183750144170591e+00],
            [0.0, 0.0, 9.466003750790984e-02, 2.133365656601689e+00]])

    Matching_End = np.array([[1.251215137881939e+00, -3.351564258427984e+01, 0.0, 0.0],
            [-3.538979594916994e-02,  1.747190939412409e+00,  0.0, 0.0],
            [0.0,  0.0, -8.577933984345419e-01, -6.305922371634640e+00],
            [0.0,  0.0, -5.462243508716558e-02, -1.567329426716162e+00]])

    T1 = np.array([9.7849, 161.9406, 0.5974058, 4.0106, 157.1455, 0.10872034]).reshape((6,1))
    T2 = np.array([-4.1204, 112.4439, 0.15988147, -0.3483, 59.0122, 0.019001374]).reshape((6,1))
    T3 = np.array([0.4231, 9.9088, 0.1189865, -3.5673, 48.4153, 0.2834977]).reshape((6,1))


    T1_end = twiss_after_transportation(Matching_Begin, T1)
    T2_end = twiss_after_transportation(Matching_End, T2)
    print(BMAG(T1_end, T3))
    print(BMAG(T2_end, T3))
