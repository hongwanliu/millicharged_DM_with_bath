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

    # function to log-Fourier transform
    ar_ary = f(r_ary) * (r_ary)**(1.5 - q)

    # log-Fourier transform result
    ak_ary = (2*np.pi)**1.5 * k_ary**(-1.5-q) * pyfftlog.fht(ar_ary.copy(), xsave, tdir)

    return (k_ary, ak_ary)






