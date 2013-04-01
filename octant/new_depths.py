import numpy as np

# Variable definition
# -------------------
#
#    N(ng) : Number of vertical levels for each nested grid.         
#                                                                    
#     zeta : time-varying free-surface, zeta(x,y,t), (m)             
#                                                                    
#        h : bathymetry, h(x,y), (m, positive, maybe time-varying)   
#                                                                    
#       hc : critical (thermocline, pycnocline) depth (m, positive)  
#                                                                    
#        z : vertical depths, z(x,y,s,t), meters, negative           
#              z_w(x,y,0:N(ng))      at   W-points  (top/bottom cell)
#              z_r(z,y,1:N(ng))      at RHO-points  (cell center)    
#                                                                    
#              z_w(x,y,0    ) = -h(x,y)                              
#              z_w(x,y,N(ng)) = zeta(x,y,t)                          
#                                                                    
#        s : nondimensional stretched vertical coordinate,           
#             -1 <= s <= 0                                           
#                                                                    
#              s = 0   at the free-surface, z(x,y, 0,t) = zeta(x,y,t)
#              s = -1  at the bottom,       z(x,y,-1,t) = - h(x,y,t) 
#                                                                    
#              sc_w(k) = (k-N(ng))/N(ng)       k=0:N,    W-points    
#              sc_r(k) = (k-N(ng)-0.5)/N(ng)   k=1:N,  RHO-points    
#                                                                    
#        C : nondimensional vertical stretching function, C(s),      
#              -1 <= C(s) <= 0                                       
#                                                                    
#              C(s) = 0    for s = 0,  at the free-surface           
#              C(s) = -1   for s = -1, at the bottom                 
#                                                                    
#              Cs_w(k) = F(s,theta_s,theta_b)  k=0:N,    W-points    
#              Cs_r(k) = C(s,theta_s,theta_b)  k=1:N,  RHO-points    
#                                                                    
#       Zo : vertical transformation functional, Zo(x,y,s):          
#                                                                    
#              Zo(x,y,s) = H(x,y)C(s)      separable functions       




#  Vtransform
#  ----------
#                                                                     
#  Vtransform == 1
#  ---------------
#
#  (1) Original transformation (Shchepetkin and McWilliams, 2005): In 
#      ROMS since 1999 (version 1.8):                                 
#
#  Original formulation: Compute vertical depths (meters, negative) at
#                        RHO- and W-points, and vertical grid
#                        thicknesses. Various stretching functions are possible.
#
#         z(x,y,s,t) = Zo + zeta(x,y,t) * [1.0 + Zo / h(x,y)]
#
#  where,
#         Zo = hc * [s(k) - C(k)] + C(k) * h(x,y)
#  or                                                            
#         Zo(x,y,s) = hc * s + [h(x,y) - hc] * C(s)                    
#  such that                                                              
#         Zo(x,y,s) = 0         for s = 0,  C(s) = 0,  at the surface  
#         Zo(x,y,s) = -h(x,y)   for s = -1, C(s) = -1, at the bottom   
#
#         z(x,y,s,t) = Zo(x,y,s) + zeta(x,y,t) * [1 + Zo(x,y,s)/h(x,y)]
#
#
#  Vtransform == 2
#  ---------------
#  (2) New transformation: In UCLA-ROMS since 2005:  
#  New formulation: Compute vertical depths (meters, negative) at
#                   RHO- and W-points, and vertical grid thicknesses.
#  Various stretching functions are possible.
#
#         z_w(x,y,s,t) = zeta(x,y,t) + [zeta(x,y,t)+ h(x,y)] * Zo_w
#
#                 Zo_w = [hc * s(k) + C(k) * h(x,y)] / [hc + h(x,y)]                 
#                                                                     
#      where                                                          
#                                                                     
#        Zo(x,y,s) = [hc * s(k) + h(x,y) * C(k)] / [hc + h(x,y)]      
#                                                                     
#        Zo(x,y,s) = 0         for s = 0,  C(s) = 0,  at the surface  
#        Zo(x,y,s) = -1        for s = -1, C(s) = -1, at the bottom   
#
#
#        z(x,y,s,t) = zeta(x,y,t) + [zeta(x,y,t) + h(x,y)] * Zo(x,y,s)
#                                                                     




def get_Vstretching(Vstretching, theta_s, theta_b, Hscale=3):
    
    if Vstretching == 1:
        
        assert 0 <  theta_s <= 8, 'theta_s not in valid range for Vstretching == 1'
        assert 0 <= theta_b <= 1, 'theta_b not in valid range for Vstretching == 1'

        def C(s):
            '''
            return C(s), the vertical coordinate stretching for Vstretching == 1
            
            C(s) = (1 - b) * [SINH(s * a) / SINH(a)] +
                       b * [-0.5 + 0.5 * TANH(a * (s + 0.5)) / TANH(0.5 * a)]
            
            where the stretching parameters (a, b) are specify at input:
            
                   a = theta_s               0 <  theta_s <= 8
                   b = theta_b               0 <= theta_b <= 1
            
            If theta_b=0, the refinement is surface intensified as theta_s is increased.
            
            If theta_b=1, the refinement is both bottom and surface intensified
            as theta_s is increased.
            
            '''
            
            a = theta_s
            b = theta_b
            
            return (1 - b) * (np.sinh(s * a) / np.sinh(a)) + \
                    b * (-0.5 + 0.5 * np.tanh(a * (s + 0.5)) / np.tanh(0.5 * a))
    
    elif Vstretching == 2:
        
        def C(s):
            '''
            return C(s), the vertical coordinate stretching for Vstretching == 2
            
            This vertical stretching function is defined, in the simplest form, as:
            
                C(s) = [1.0 - COSH(theta_s * s)] / [COSH(theta_s) - 1.0]
            
            it is similar in meaning to the original vertical stretcing function
            (Song and Haidvogel, 1994), but note that hyperbolic functions are
            COSH, and not SINH.
            
            Note that the above definition results in
                   -1 <= C(s) <= 0
            as long as
                   -1 <= s <= 0
            and, unlike in any previous definition
                   d[C(s)]/ds  -->  0      if  s -->  0
            For the purpose of bottom boundary layer C(s) is further modified
            to allow near-bottom refinement.  This is done by blending it with
            another function.
            '''
            
            return (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
    
    elif Vstretching == 3:
        
        assert 0.0 <  theta_s , 'theta_s must be positive for Vstretching == 3'
        assert 0.0 <= theta_b , 'theta_b must be non-negative for Vstretching == 3'
        
        def C(s):
            '''
            This stretching function is intended for very shallow coastal
            applications, like gravity sediment flows.
            
            At the surface, C(s=0)=0
            
                C(s) = - LOG(COSH(Hscale * ABS(s) ** alpha)) /
                         LOG(COSH(Hscale))
            
            At the bottom, C(s=-1)=-1
            
                C(s) = LOG(COSH(Hscale * (s + 1) ** beta)) /
                       LOG(COSH(Hscale)) - 1
            
            where
            
                 Hscale : scale value for all hyperbolic functions
                            Hscale = 3.0    set internally here
                  alpha : surface stretching exponent
                            alpha = 0.65   minimal increase of surface resolution
                                    1.0    significant amplification
                   beta : bottoom stretching exponent
                            beta  = 0.58   no amplification
                                    1.0    significant amplification
                                    3.0    super-high bottom resolution
                      s : stretched vertical coordinate, -1 <= s <= 0
                            s(k) = (k-N)/N       k=0:N,    W-points  (s_w)
                            s(k) = (k-N-0.5)/N   k=1:N,  RHO-points  (s_rho)
            
            The stretching exponents (alpha, beta) are specify at input:
            
                   alpha = theta_s
                   beta  = theta_b
            
            '''
            exp_sur = theta_s
            exp_bot = theta_b
            
            Cbot = np.log(np.cosh(Hscale * (s + 1.0)**exp_bot))/np.log(np.cosh(Hscale)) - 1.0
            Csur = -np.log(np.cosh(Hscale * np.abs(s)**exp_sur))/np.log(np.cosh(Hscale))
            Cweight = 0.5 * (1.0 - np.tanh(Hscale * (s + 0.5)))
            
            return Cweight * Cbot + (1.0 - Cweight) * Csur
    
    elif Vstretching == 4:
        
        assert 0.0 <  theta_s <= 10.0, 'theta_s not in valid range for Vstretching == 4'
        assert 0.0 <= theta_b <= 3.0, 'theta_s not in valid range for Vstretching == 4'
        
        def C(s):
            '''
            The range of meaningful values for the control parameters are:
        
                 0 <  theta_s <= 10.0
                 0 <= theta_b <=  3.0
        
            Users need to pay attention to extreme r-factor (rx1) values near
            the bottom.
        
            This vertical stretching function is defined, in the simplest form,
            as:
        
                C(s) = [1.0 - COSH(theta_s * s)] / [COSH(theta_s) - 1.0]
        
            it is similar in meaning to the original vertical stretcing function
            (Song and Haidvogel, 1994), but note that hyperbolic functions are
            COSH, and not SINH.
        
            Note that the above definition results in
        
                   -1 <= C(s) <= 0
        
            as long as
        
                   -1 <= s <= 0
        
            and
        
                   d[C(s)]/ds  -->  0      if  s -->  0
        
            For the purpose of bottom boundary layer C(s) is further modified
            to allow near-bottom refinement by using a continuous, second
            stretching function
        
                   C(s) = [EXP(theta_b * C(s)) - 1.0] / [1.0 - EXP(-theta_b)]
        
            This double transformation is continuous with respect to "theta_s"
            and "theta_b", as both values approach to zero.
            '''
            C = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
            return (np.exp(theta_b * C) - 1.0) / (1.0 - np.exp(-theta_b))
    
    else:
        
        raise Exception('Vstretching must be 1, 2, 3, or 4')
                       
    return C

def get_depths(Vtransform, C, h, hc):
    '''
    return vertical transformation function, Zo(s), that returns the depths when zeta == 0
    
    Inputs
    ------
    
    Vtransform : integer
            Either 1 or 2, specifies the vertical transform equation
    C : function
            Function C(s) returns the vertical stretching.  Obtained with get_Vstretching()
    h : scalar, 1D, or 2D array
            The water depth(s) at which to calculate the depths
    hc : float
            The critical depth that defines regions of enhanced vertical resolution
    
    Output
    ------
    depths : function
            The function depths(s) returns the depths with no free surface anomalies (zeta == 0)
    '''
    
    if Vtransform == 1:
        def depths(s, zeta=0):
            '''
            return depths(s, zeta=0) based on Vtransform == 1
            
            Original transformation (Shchepetkin and McWilliams, 2005): In 
            ROMS since 1999 (version 1.8):                                 
            
            Original formulation: Compute vertical depths (meters, negative) at
                                  RHO- and W-points, and vertical grid
                                  thicknesses. Various stretching functions are possible.
            
                   depths(x,y,s,t) = Zo(x,y,s) + zeta(x,y,t) * [1 + Zo(x,y,s)/h(x,y)]
            
            where,
                   Zo = hc * [s(k) - C(k)] + C(k) * h(x,y)
            or                                                            
                   Zo(x,y,s) = hc * s + [h(x,y) - hc] * C(s)                    
            such that                                                              
                   Zo(x,y,s) = 0         for s = 0,  C(s) = 0,  at the surface  
                   Zo(x,y,s) = -h(x,y)   for s = -1, C(s) = -1, at the bottom   
            
            The size of h defines the horizontal dimensions of the depths.
            Zeta must be broadcastable over h; the rightmost dimensions of zeta
            must be the dimensions of h.  Zeta may have at most one additional
            dimension (to the left), which represents time.
            
            '''
            if np.ndim(h) == 1:
                s = s[:, np.newaxis]
            if np.ndim(h) == 2:
                s = s[:, np.newaxis, np.newaxis]
            
            if zeta.ndim > h.ndim:
                zeta[:, np.newaxis, ...]
            
            assert hc <= np.min(h), 'hc cannot be larger than the minimum depth'
            
            def Zo(s):
                return hc * (s - C(s)) + C(s) * h
            
            return Zo(s) + zeta * (1 + Zo(s)/h)
        
        return depths
            
    elif Vtransform == 2:

        def depths(s, zeta=0):
            '''
            return depths(s, zeta=0) based on Vtransform == 2
            
            New transformation: In UCLA-ROMS since 2005.
            New formulation: Compute vertical depths (meters, negative) at
                             RHO- and W-points, and vertical grid thicknesses.
            Various stretching functions are possible.
            
                    depths(x,y,s,t) = zeta(x,y,t) + [zeta(x,y,t)+ h(x,y)] * Zo(x, y, s)
            
            where                                                          
                    Zo = [hc * s(k) + C(k) * h(x,y)] / [hc + h(x,y)]                 
            or                                                                   
                    Zo(x,y,s) = [hc * s(k) + h(x,y) * C(k)] / [hc + h(x,y)]      
            such that                                                             
                    Zo(x,y,s) = 0         for s = 0,  C(s) = 0,  at the surface  
                    Zo(x,y,s) = -1        for s = -1, C(s) = -1, at the bottom   
            
            The size of h defines the horizontal dimensions of the depths.
            Zeta must be broadcastable over h; the rightmost dimensions of zeta
            must be the dimensions of h.  Zeta may have at most one additional
            dimension (to the left), which represents time.

            '''
            if np.ndim(h) == 1:
                s = s[:, np.newaxis]
            if np.ndim(h) == 2:
                s = s[:, np.newaxis, np.newaxis]
                        
            def Zo(s):
                return (hc * s + C(s) * h) / (hc + h)  
            
            return zeta + (zeta + h) * Zo(s)
        
        return depths
            

def get_sw(N):
    return np.linspace(-1.0, 0.0, N)

def get_srho(N):
    sr = np.linspace(-1.0, 0.0, N+1)
    return 0.5 * (sr[1:] + sr[:-1])

# class s_coordinate(object):
#     """return an object that can be indexed to return depths
#     
#     Parameters
#     ----------
#     h : array_like
#         An array of depths, either one- or two-dimensional.
#     hc : float
#         The critical depth.  Presently hc < h.min() must be true.
#     theta_b : float
#         A parameter (0.0 < theta_b < 1.0) that says whether the coordinate
#         will be focused at the surface (theta_b -> 1.0) or split evenly
#         between surface and bottom (theta_b -> 0).
#     theta_s : float
#         A parameter (typically 0.0 <= theta_s < 5.0) that defines the amount
#         of grid focising. A higher value for theta_s will focus the grid more.
#     N : int
#         The number of rho/tracer-points in the vertical.
#     grid:  'rho' or 'w'
#         Output values at the vertical 'rho' or vertical 'w' points.
#     zeta: array_like, optional
#         The free surface, must be the same size as h.  Defaults to zeta=0.
#     Vtransform:  1 or 2
#          
#     Vstretching:  1, 2, 3, or 4
#     
#     Returns
#     -------
#     z : object
#         z is an scoord object that contains as attributes all of the input
#         values. The actual depths may be retreived by indexing z. Note that
#         the values of zeta are not calculated until z is indexed, so a netCDF
#         variable for zeta may be passed, even if the file is large, as only
#         the values that are required will be retrieved from the file.
#     
#     """
#     def __init__(self, h, hc, theta_b, theta_s, N, grid, zeta=None, Vtransform=1, Vstretching=1):
#         self.hc = hc
#         self.h = np.asarray(h)
#         self.theta_b = theta_b
#         self.theta_s = theta_s
#         self.N = int(N)
#         self.grid = grid
#         
#         if self.grid == 'w':
#             self.N += 1
#         
#         if zeta is None:
#             self.zeta = np.zeros(h.shape)
#         else:
#             self.zeta = zeta
#         
#         hdim = len(h.shape)
# 
#     def _get_sc(self):
#         if self.grid == 'rho':
#             sc = np.linspace(-1.0, 0.0, self.N+1)
#             sc = 0.5*(sc[1:]+sc[:-1])
#         else:
#             sc = np.linspace(-1.0, 0.0, self.N)
#         return sc
#     
#     sc = property(_get_sc)
#     
#     def _get_Cs(self):
#         return (1-self.theta_b)*np.sinh(self.theta_s*self.sc)/np.sinh(self.theta_s) \
#                 + 0.5*self.theta_b*(np.tanh( self.theta_s*(self.sc+0.5))- \
#                                              np.tanh(0.5*self.theta_s ))  \
#                                   /np.tanh(0.5*self.theta_s)
#     
#     Cs = property(_get_Cs)
#     
#     def __getitem__(self, key):
#         if isinstance(key, tuple) and len(self.zeta.shape) > len(self.h.shape):
#             zeta = self.zeta[key[0]]
#             res_index = (slice(None),) + key[1:]
#         elif len(self.zeta.shape) > len(self.h.shape):
#             zeta = self.zeta[key]
#             res_index = slice(None)
#         else:
#             zeta = self.zeta
#             res_index = key
#         
#         if self.h.ndim == zeta.ndim:       # Assure a time-dimension exists
#             zeta = zeta[np.newaxis, :]
#         
#         ti = zeta.shape[0]
#         z = np.empty((ti, self.N) + self.h.shape, 'd')
#         for n in range(ti):
#             for  k in range(self.N):
#                 z0=(self.sc[k]-self.Cs[k])*self.hc + self.Cs[k]*self.h;
#                 z[n,k,:] = z0 + zeta[n,:]*(1.0 + z0/self.h);
#         
#         return np.squeeze(z[res_index])


if __name__ == '__main__':
    import netCDF4
    import matplotlib.pyplot as plt
    
    plot_Cs = True
    
    s = np.linspace(-1, 0, 30)
    
    
    C1 = get_Vstretching(1, theta_s=5.0, theta_b=0.5)
    C2 = get_Vstretching(2, theta_s=5.0, theta_b=0.5)
    C3 = get_Vstretching(3, theta_s=1.0, theta_b=1.0)
    C4 = get_Vstretching(4, theta_s=5.0, theta_b=1.0)
    
    if plot_Cs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(s, C1(s), '-r')
        ax.plot(s, C2(s), '-b')
        ax.plot(s, C3(s), '-m')
        ax.plot(s, C4(s), '-g')
        plt.legend(['Vstretching 1', 'Vstretching 2', 'Vstretching 3', 'Vstretching 4'], loc='lower right')
        ax.set_xlabel('s')
        ax.set_ylabel('C(s)')
        plt.show()
    
    h = np.linspace(10, 100, 50)
    hc = 3
    
    # depths = get_depths(Vtransform=2, C=C1, h=h, hc=hc)
    
    # nc = netCDF4.Dataset('ocean_his_01.nc')
    # z = nc_depths(grid='w', nc=nc)
    # 
    # print z[5].shape
    # print z[1, 2].shape
    # print z[1, 2, 3].shape
    # print z[1, 2, 3, 4].shape

    
    
    
    