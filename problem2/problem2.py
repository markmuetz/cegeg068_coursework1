from __future__ import division

import numpy as np
import pylab as plt

import distmesh as dm

def rectangle_with_circle_at_corner():
    # Does not terminate!
    fd = lambda p: dm.ddiff(dm.drectangle(p,0,10,0,6), dm.dcircle(p,10,0,4))
    return dm.distmesh2d(fd, dm.huniform, 0.9, (0, 0, 10, 6))

def rectangle_with_circle_at_centre():
    # Does not terminate!
    fd = lambda p: dm.ddiff(dm.drectangle(p,0,20,0,12), dm.dcircle(p,10,6,4))
    return dm.distmesh2d(fd, dm.huniform, 0.5, (0, 0, 20, 12), max_iter=100)

def simple_rect():
    fd = lambda p: dm.drectangle(p,0,1,0,1)
    return dm.distmesh2d(fd, dm.huniform, 0.1, (0, 0, 1, 1), max_iter=100)

def make_C():
    C = np.array([[1, 2, 4, 3],
                  [2, 5, 4, 0],
                  [3, 4, 8, 7],
                  [4, 5, 8, 0],
                  [5, 6, 8, 0],
                  [6, 9, 8, 0]])
    return C

def make_Ks():
    Ks = []
    Ks.append((1/60) *np.array([[ 58,  17, -29, -46],
                                [ 17,  58, -46, -29],
                                [-29, -46,  58, -17],
                                [-46, -29,  17,  58]]))
    Ks.append((1/2)  *np.array([[ 1,  0, -1],
                                [ 0,  1, -1],
                                [-1, -1,  2]]))
    Ks.append((1/240)*np.array([[164, -14, -82, -68],
                                [-14, 164, -68, -82],
                                [-82, -68, 164, -14],
                                [-68, -82, -14,  164]]))
    Ks.append((1/4)  *np.array([[ 5, -4, -1],
                                [-4,  4,  0],
                                [-1,  0,  1]]))
    Ks.append((1/32) *np.array([[ 29, -18, -11],
                                [-18,  20, -2 ],
                                [-11,  -2,  13]]))
    Ks.append((1/20) *np.array([[ 25, -25,  0],
                                [-25,  29, -4],
                                [  0,  -4,  4]]))
    return Ks

def construct_K_string(C):
    '''Prints stiffness matrix elements from a connectivity matrix'''
    K = np.zeros((np.max(C), np.max(C))).astype(object)
    for i in range(len(C)):
        for j in range(len(C[i])):
            for k in range(len(C[i])):
                Kstr = 'K%d(%d,%d)'%(i + 1, j + 1, k + 1)
                if C[i, j] == 0 or C[i, k] == 0:
                    continue

                if K[C[i, j] - 1, C[i, k] - 1] == 0:
                    K[C[i, j] - 1, C[i, k] - 1] = Kstr
                else:
                    K[C[i, j] - 1, C[i, k] - 1] += '+' + Kstr
    return K

def construct_K(C, Ks):
    '''Constructs a stiffness matrix from a connectivity matrix'''
    K = np.zeros((np.max(C), np.max(C)))
    for i in range(len(C)):
        for j in range(len(C[i])):
            for k in range(len(C[i])):
                Kstr = 'K%d(%d,%d)'%(i + 1, j + 1, k + 1)
                if C[i, j] == 0 or C[i, k] == 0:
                    continue

                if K[C[i, j] - 1, C[i, k] - 1] == 0:
                    K[C[i, j] - 1, C[i, k] - 1] = Ks[i][j, k]
                else:
                    K[C[i, j] - 1, C[i, k] - 1] += Ks[i][j, k]
    return np.matrix(K)

def run_project():
    #plt.ion()
    #return rectangle_with_circle_at_corner()
    #return simple_rect()
    #return rectangle_with_circle_at_centre()
    C = make_C()
    Ks = make_Ks()

    K_str = construct_K_string(C)
    K = construct_K(C, Ks)

    for i in [3, 4]:
        for j in [3, 4, 7, 8, 9]:
            print("K(%d, %d) : %s = %f"%(i, j, K_str[i - 1, j - 1], K[i - 1, j - 1]))

    Kp = np.matrix([[K[2, 2], K[2, 3]],
                    [K[3, 2], K[3, 3]]])
    up = np.matrix([K[2, 6] + K[2, 7] + K[2, 8], K[3, 6] + K[3, 7] + K[3, 8]]).T

    print(Kp)
    print(up)
    print(-Kp.I*up)

    #print(K_str)
    #print(K)


if __name__ == "__main__":
    run_project()
