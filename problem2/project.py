from __future__ import division

import numpy as np
#import pylab as plt

import distmesh as dm

def rectangle_with_circle_at_corner():
    # Does not terminate!
    fd = lambda p: dm.ddiff(dm.drectangle(p,0,10,0,6), dm.dcircle(p,10,0,4))
    return dm.distmesh2d(fd, dm.huniform, 0.9, (0, 0, 10, 6))

def rectangle_with_circle_at_centre():
    # Does not terminate!
    fd = lambda p: dm.ddiff(dm.drectangle(p,0,20,0,12), dm.dcircle(p,10,6,4))
    return dm.distmesh2d(fd, dm.huniform, 0.5, (0, 0, 20, 12))

def run_project():
    #return rectangle_with_circle_at_corner()
    return rectangle_with_circle_at_centre()

if __name__ == "__main__":
    run_project()
