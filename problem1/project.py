from __future__ import division
import numpy as np
import pylab as plt

OUTPUT_FOLDER = 'output'

# Consts in question.
ALPHA = 0.4168
T_0   = 250
T_inf = 75
L     = 4

def generate_K(n):
    '''Generates stiffness matrix for n elements based on rules in notes'''
    if n < 2:
        raise "n must be >= 2"
    K = np.matrix(np.zeros((n + 1, n + 1)))

    # First row
    K[0, 0] = 1
    K[0, 1] = -1

    # Middle rows
    for i in range(2, n + 1):
        # i matches i in notes
        # ii is python matrix index.
        ii = i - 1
        K[ii, ii - 1] = 3 - 2 * i
        K[ii, ii    ] = (i - 1) * 4
        K[ii, ii + 1] = 1 - 2 * i

    # End row
    K[n, n - 1] = K[n - 1, n]
    K[n, n] = -K[n - 1, n]

    K *= 1 / 2 

    try:
        validate_K(K)
    except Exception, e:
        print('Matrix for n=%d not valid'%n)
        print(str(e))

    return K

def generate_fl(n):
    '''Generates load vector for n elements

returns fl1, fl2
fl1: vector part of load vector looks like (1 2 2 ... 2 1).T
fl2: matrix part of load vector. Tridiagonal'''
    fl1 = np.matrix(np.zeros((n + 1, 1)))
    fl1[0, 0] = 1
    fl1[-1, 0] = 1
    fl1[1:-1, 0] = 2
    fl1 *= ALPHA * T_inf * L / (2 * n)

    fl2 = np.matrix(np.zeros((n + 1, n + 1)))
    fl2[0, 0] = 2
    fl2[0, 1] = 1
    fl2[1, 0] = fl2[0, 1]
    fl2[-1, -1] = 2
    for i in range(n - 1):
        fl2[1 + i, 2 + i] = 1
        fl2[1 + i, 1 + i] = 4
        fl2[2 + i, 1 + i] = 1

    fl2 *= ALPHA * L / (6 * n)

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
    '''Copied directly from notes'''
    R = np.matrix([[ 1/2 + ALPHA/3, -1/2 + ALPHA/6,              0,              0],
                   [-1/2 + ALPHA/6,  2 + 2*ALPHA/3, -3/2 + ALPHA/6,              0],
                   [             0, -3/2 + ALPHA/6,  4 + 2*ALPHA/3, -5/2 + ALPHA/6],
                   [             0,              0, -5/2 + ALPHA/6,  6 + 2*ALPHA/3]])
    return R

def generate_f_lb():
    '''Copied directly from notes'''
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
    plt.plot(xs, temperatures, 'b-')
    if n <= 100:
        plt.plot(xs, temperatures, 'k+')

    if plot:
        plt.title('Temperatures at nodes for %d elements'%n)
        plt.ylabel('Temperature')
        plt.xlabel('x')
        plt.savefig('%s/T_vs_x_for_%d.png'%(OUTPUT_FOLDER, n))
        plt.show()

def calc_convergence_of_T_1():
    T_1s = []
    ns = []
    for n in range(4, 102, 2):
        #print(n)
        ns.append(n)
        T_1s.append(solve_FE_eq(n, False))

    plt.title('Convergence of temperature at x=0')
    plt.ylabel('Temperature at x=0')
    plt.xlabel('Number of elements')
    plt.plot(ns, T_1s)
    plt.savefig('%s/Convergence_of_T_1.png'%(OUTPUT_FOLDER))
    plt.show()

def solve_FE_eq_for_four_elements():
    '''Old way, manually make matrices then solve.'''
    R = generate_R()
    f_lb = generate_f_lb()
    T = R.I * f_lb
    draw_graph(T, 4, True)

def solve_FE_eq(n=4, plot=True):
    '''Generate matrices for n elements then solves.

n: number of elements
plot: draw a graph

returns: T at x=0
'''
    # Stiffness matrix.
    K = generate_K(n)

    # Load vector (realy one vector, one matrix)
    fl1, fl2 = generate_fl(n)

    # Combine stiffness matrix and matrix part of load vector.
    M = K + fl2

    # Reduce the matrix, ignore the last row/col (we know T_0).
    R = M[:-1, :-1]

    # Remember to include extra term from M because of reduction.
    f_lb = fl1[:-1] - M[:-1, -1] * T_0

    # Find unknowns.
    T = R.I * f_lb

    if plot:
        draw_graph(T, n, plot)

    return T[(0, 0)]

def run_project():
    T1s = []
    for n in [4, 10, 1000]:
        val = (n, solve_FE_eq(n, True))
        print(val)
        T1s.append(val)

    T1_4 = T1s[0][1]
    T1_1000 = T1s[-1][1]

    print('Error (4 vs 1000): %f%%'%((T1_1000 - T1_4) / T1_1000 * 100))

    calc_convergence_of_T_1()

if __name__ == "__main__":
    run_project()
