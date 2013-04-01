# encoding: utf-8
"""Tools for calculating vertical velocity in s-coordinate models."""
__docformat__ = "restructuredtext en"

import numpy as np

def omega(u, v, pm, pn, Hz, detadt=0.):
    '''
    Calculate omega, the grid-relative vertical velocity, on the s-coordinate.
    By definition, omega(sigma=-1) = omega(sigma=0) = 0.
    
    Parameters
    ----------
    u, v : 3D or 4D array
        The horizontal velocity values, on their respective CGrid points
    pm, pn : 2D array
        The grid metrics, inverse grid widths
    Hz : 3D or 4D array
        The vertical layer thickness at rho-points
    detadt : 2D or 4D array
        The rate of change of the sea-level.  [Default = 0]
        
    Returns
    -------
    omega : 3D array
        The grid-relative vertical velocity on w-points
            
    '''
    
    D = Hz.sum(axis=-3)
    
    Hzom = (Hz[...,:-1] + Hz[...,1:]) / (pm[:,:-1] + pm[:,1:])
    Hzon = (Hz[...,:-1,:] + Hz[...,1:,:]) / (pn[:-1,:] + pn[1:,:])
    
    utrans = Hzom * u / pm
    vtrans = Hz_v * v