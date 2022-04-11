"""Logarithmic FFT, based entirely on pyfftlog
"""

import numpy as np
import pyfftlog 

def fftj0(f, logrmin, logrmax, n_pts=4096, q=0):
    """Fourier transform of function a(r). 

    The actual integral computed is \int d^3 r a(r) j_0(k r), which is the Fourier transform for a function that only depends on magnitude of r. 

    Parameters
    ----------
    f : function 
        function to FFT, returns an array of function values. 
    logrmin : float
        log10 minimum value of r to include. 
    logrmax : float
        log10 maximum value of r to include. 
    n_pts : int, optional
        number of data points to use, max = 4096. 
    q : float, optional
        the bias of the integral to use 

    Returns
    -------
    tuple of ndarray
        Returns k abscissa and result. 

    Notes
    -------

    pyfftlog will evaluate \int dr k (kr)^q J_1/2(kr) a(r) (kr)^(3/2 - q), and the bias q can be set arbitrarily, although q = 0 usually gives the best performance. 
    """

    # Sensible approximate choice of k_c r_c
    kr = 1

    # Tell fhti to change kr to low-ringing value
    # WARNING: kropt = 3 will fail, as interaction is not supported
    kropt = 1

    # Forward transform (changed from dir to tdir, as dir is a python fct)
    tdir = 1

    # Central point log10(r_c) of periodic interval
    logrc = (logrmin + logrmax)/2

    # Central index (1/2 integral if n is even)
    nc = (n_pts + 1)/2.0

    # Log-spacing of points
    dlogr = (logrmax - logrmin)/n_pts
    dlnr = dlogr*np.log(10.0)


    # Initialization. kr = k_c r_c, where c is the central point. 
    kr, xsave = pyfftlog.fhti(n_pts, 0.5, dlnr, q, kr, kropt)
    logkc = np.log10(kr) - logrc


    # Actual r-binning
    r_ary = 10**(logrc + (np.arange(1, n_pts+1) - nc)*dlogr)
    # Actual k-binning
    k_ary = 10**(logkc + (np.arange(1, n_pts+1) - nc)*dlogr)

    # function to log-Fourier transform. 
    # In general f returns something multidimensional. 
    # ar_ary has dimensions ... x r_ary. 
    ar_ary = np.moveaxis(f(r_ary), 0, -1) * (r_ary)**(1.5 - q)

    # dimensions ... x r_ary
    ak_ary = np.zeros(ar_ary.shape) 

    if len(ak_ary.shape) > 1: 
    
        # Array of indices, dimensions ... x 2
        indices_ary = np.moveaxis(np.indices(ak_ary[...,0].shape), 0, -1)

        for ind in indices_ary.reshape(-1, indices_ary.shape[-1]): 

            if ind.shape == ():

                ak_ary[ind] = (2*np.pi)**1.5 * k_ary**(-1.5-q) * pyfftlog.fht(ar_ary[ind].copy(), xsave, tdir)

            else:

                ak_ary[tuple(ind)] = (2*np.pi)**1.5 * k_ary**(-1.5-q) * pyfftlog.fht(ar_ary[tuple(ind)].copy(), xsave, tdir)

    else: 

        ak_ary = (2*np.pi)**1.5 * k_ary**(-1.5-q) * pyfftlog.fht(ar_ary.copy(), xsave, tdir)

    # Return as dimension ... x k_ary
    return (k_ary, ak_ary)






