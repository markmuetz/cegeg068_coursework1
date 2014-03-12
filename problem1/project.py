from __future__ import division
import numpy as np
import pylab as plt

ALPHA = 0.4168
T_0   = 250
T_inf = 75
L     = 4

def generate_K(n):
    if n < 2:
        raise "n must be >= 2"
    K = np.matrix(np.zeros((n + 1, n + 1)))
    K[0, 0] = 1
    K[0, 1] = -1
    K[1, 0] = K[0, 1]
    for i in range(n - 1):
        K[1 + i, 1 + i] = (i + 1) * 4
        K[1 + i, 2 + i] = -3 - 2 * i
        K[2 + i, 1 + i] = K[1 + i, 2 + i] 
    K[n - 1, n] = -((n + 1) * 2 - 3)
    K[n, n - 1] = K[n - 1, n]
    K[n, n] = -K[n - 1, n]
    K *= 1 / 2 
    #validate_K(K)
    return K

def generate_fl(n):
    alpha = L * ALPHA / n
    fl1 = np.matrix(np.zeros((n + 1, 1)))
    fl1[0, 0] = 1
    fl1[-1, 0] = 1
    fl1[1:-1, 0] = 2
    fl1 *= alpha * T_inf / 2

    fl2 = np.matrix(np.zeros((n + 1, n + 1)))
    fl2[0, 0] = 2
    fl2[0, 1] = 1
    fl2[1, 0] = fl2[0, 1]
    fl2[-1, -1] = 2
    for i in range(n - 1):
        fl2[1 + i, 2 + i] = 1
        fl2[2 + i, 1 + i] = 1
        fl2[1 + i, 1 + i] = 4

    fl2 *= alpha / 6

    return fl1, fl2

def validate_K(K):
    if K.shape[0] != K.shape[1]:
        raise Exception("K is not square")
    # This test causes problems for some reason.
    #if np.linalg.det(K) != 0:
        #raise Exception("det(K) != 0")
    for i in range(len(K)):
        if K[i].sum() != 0:
            raise Exception("row %i != 0"%i)
        if K[:, i].sum() != 0:
            raise Exception("col %i != 0"%i)
        for j in range(i):
            if K[i, j] != K[j, i]:
                raise Exception("matrix not symmetrix")

def generate_R():
    R = np.matrix([[ 1/2 + ALPHA/3, -1/2 + ALPHA/6,              0,              0],
                   [-1/2 + ALPHA/6,  2 + 2*ALPHA/3, -3/2 + ALPHA/6,              0],
                   [             0, -3/2 + ALPHA/6,  4 + 2*ALPHA/3, -5/2 + ALPHA/6],
                   [             0,              0, -5/2 + ALPHA/6,  6 + 2*ALPHA/3]])
    return R

def generate_f_lb():
    f_lb = np.matrix([[ALPHA * T_inf / 2],
                      [ALPHA * T_inf],
                      [ALPHA * T_inf],
                      [ALPHA * T_inf - (ALPHA/6 - 7/2) * T_0]])
    return f_lb



def multi_run(max_n):
    for i in range(2, max_n):
        run_project(i, False)
    plt.show()

def draw_graph(T, n, plot):
    temperatures = [T[(i, 0)] for i in range(len(T))]
    temperatures.append(250)
    xs = np.arange(len(temperatures)) * 1.
    xs *= 4. / n
    plt.plot(xs, temperatures)
    plt.plot(xs, temperatures, 'k+')
    if plot:
        plt.show()

def run_project(n=4, plot=True):
    Rs = []
    if False:
        # Old way
        R = generate_R()
        f_lb = generate_f_lb()
        T = R.I * f_lb
        #! T_1 == T[0] to translate between code and question.
        draw_graph(T, 4, plot)
        Rs.append(R)
    
    if True:
        # New way
        K = generate_K(n)
        fl1, fl2 = generate_fl(n)

        M = K + fl2

        f_lb = fl1[:-1] - M[:-1, -1] * T_0
        R = M[:-1, :-1]

        T = R.I * f_lb
        draw_graph(T, n, plot)
        Rs.append(M)

    return Rs

if __name__ == "__main__":
    run_project()
