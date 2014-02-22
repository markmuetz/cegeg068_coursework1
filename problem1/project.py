from __future__ import division
import numpy as np
import pylab as plt

ALPHA = 0.4168
T_0   = 250
T_inf = 75
L     = 4

def generate_M():
    M = np.matrix([[ 1/2 + ALPHA/3, -1/2 + ALPHA/6,              0,              0],
                  [-1/2 + ALPHA/6,  2 + 2*ALPHA/3, -3/2 + ALPHA/6,              0],
                  [             0, -3/2 + ALPHA/6,  4 + 2*ALPHA/3, -5/2 + ALPHA/6],
                  [             0,              0, -5/2 + ALPHA/6,  6 + 2*ALPHA/3]])
    return M

def generate_f_lb():
    f_lb = np.matrix([[ALPHA * T_inf / 2],
                      [ALPHA * T_inf],
                      [ALPHA * T_inf],
                      [ALPHA * T_inf - (ALPHA/6 - 7/2) * T_0]])
    return f_lb



def run_project():
    M = generate_M()
    f_lb = generate_f_lb()
    T = M.I * f_lb
    #! T_1 == T[0] to translate between code and question.
    temperatures = [T[(i, 0)] for i in range(len(T))]
    temperatures.append(250)
    xs = [0, 1, 2, 3, 4]
    plt.plot(xs, temperatures)
    plt.show()






    return T

if __name__ == "__main__":
    run_project()
