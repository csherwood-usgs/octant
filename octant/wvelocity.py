# encoding: utf-8
"""Tools for calculating vertical velocity in s-coordinate models."""
__docformat__ = "restructuredtext en"

import numpy as np

def omega(u, v, pm, pn, dHzdt):
    '''
    Calculate omega, the grid-relative vertical velocity, on the s-coordinate.
    By definition, omega(sigma=-1) = omega(sigma=0) = 0.
    
    Parameters
    ----------
    u, v : 3D array
        The horizontal velocity values, on their respective CGrid points
    pm, pn : 2D array
        The grid metrics, inverse grid widths
    dHzdt : 2D array
        The time rate of change of the vertical coordinate stretching parameter.
        For a standard s-coordinate, this is equivalent to 1/(h+eta) deta/dt, the
        rate of change of the sea-level divided by the total depth.
        
    Returns
    -------
    omega : 3D array
        The grid-relative vertical velocity on w-points
            
    '''