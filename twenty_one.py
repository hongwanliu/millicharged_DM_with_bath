"""21-cm physics
"""

import numpy as np
from scipy.interpolate import interp1d 

import physics as phys


omega_b_h = phys.omega_baryon * phys.h 

def xA_eff(z_ary, xA_ary, Tb_ary): 
    """Effective Lyman-alpha coupling. 
    
    See 1605.04357 Eq. (57) for complete definition. 
    
    Parameters
    ----------
    z_ary : ndarray
        Array of redshifts z. 
    xA_ary : ndarray
        Array of Lyman-alpha couplings, dimensions ... x z_ary. 
    Tb_ary : ndarray
        Array of baryon temperatures in K, dimensions ... x z_ary. 
    Returns
    -------
    ndarray

    """
    
    T_se   = 0.402 # in K

    return xA_ary / (1 + T_se / Tb_ary) * np.exp(
        -2.06 * (omega_b_h / 0.0327)**(1/3) * (phys.omega_m / 0.307)**(-1/6) 
        * np.sqrt((1. + z_ary) / 10) * (Tb_ary / T_se)**(-2/3)
    )

def xc(z_ary, Tb_ary): 
    """Collisional coupling. 

    See 1605.04357 Eq. (46), and Zygelman, ApJ 622(1356), 2005. 

    Parameters
    ----------
    z_ary : ndarray
        Array of redshifts z. 
    Tb_ary : ndarray
        Array of baryon temperatures in K, dimensions ... x z_ary.
    
    Returns
    -------
    ndarray
    """

    T_star = 0.0682 # in K, equivalent temperature of hyperfine splitting
    A10 = 2.85e-15 # in s^-1, spontaneous decay rate of hyperfine transition

    # in K
    T_kappa_10_ary = np.array([
        1., 2., 4., 6., 8., 10., 15., 20., 25., 30., 
        40., 50., 60., 70., 80., 90., 100., 200., 300.
    ])

    # in cm^3 / sec
    kappa_10_ary = np.array([
        1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12, 9.10e-12, 
        1.78e-11, 2.73e-11, 3.67e-11, 5.38e-11, 6.86e-11, 8.14e-11, 9.25e-11, 
        1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10, 2.09e-10
    ])

    log10_kappa_10_int = interp1d(
        T_kappa_10_ary, np.log10(kappa_10_ary), 
        bounds_error=False, fill_value=(np.log10(1.38e-13), np.log10(2.09e-10))
    )

    return ( 
        4 * 10**log10_kappa_10_int(Tb_ary) * phys.nH * (1. + z_ary) ** 3 
        * T_star / 3. / A10 / (phys.TCMB(1. + z_ary) / phys.kB)
    )

def TS(z_ary, xA_ary, Tb_ary, TS_equal_Tb=False): 
    """Spin temperature.  

    Parameters
    ----------
    z_ary : ndarray
        Array of redshifts z. 
    xA_ary : ndarray
        Array of Lyman-alpha couplings, dimensions ... x z_ary. 
    Tb_ary : ndarray
        Array of baryon temperatures in K, dimensions ... x z_ary. 
    TS_equal_Tb : bool
        If True, assumes TS = Tb. 
    
    Returns
    -------
    ndarray
        Spin temperature in K. 
    """

    if TS_equal_Tb: 

        return Tb_ary 

    else: 

        xA_eff_ary = xA_eff(z_ary, xA_ary, Tb_ary) 

    xc_ary = xc(z_ary, Tb_ary) 

    x_tot_ary = xA_eff_ary + xc_ary * 0 
    
    T_CMB_ary = phys.TCMB(1. + z_ary) / phys.kB

    if not TS_equal_Tb:
        TS_to_Tb_fac = x_tot_ary / (1. + x_tot_ary)
    else: 
        TS_to_Tb_fac = 1. 

    # Spin temperature
    return T_CMB_ary / (1. - TS_to_Tb_fac * (1. - T_CMB_ary / Tb_ary))
    # return (1. + x_tot_ary) / (1. / T_CMB_ary + x_tot_ary / Tb_ary)



def T21(z_ary, xA_ary, Tb_ary, TS_equal_Tb=False): 
    """21-cm brightness temperature.  

    Parameters
    ----------
    z_ary : ndarray
        Array of redshifts z. 
    xA_ary : ndarray
        Array of Lyman-alpha couplings, dimensions ... x z_ary. 
    Tb_ary : ndarray
        Array of baryon temperatures in K, dimensions ... x z_ary. 
    TS_equal_Tb : bool
        If True, assumes TS = Tb. 
    
    Returns
    -------
    ndarray
        21-cm brightness temperature in K. 
    """

    T_CMB_ary = phys.TCMB(1. + z_ary) / phys.kB

    T_s_ary = TS(z_ary, xA_ary, Tb_ary, TS_equal_Tb=TS_equal_Tb)

    # Optical depth: 1605.04357 Eq. (39)
    # tau_ary = (
    #     9.85e-3 * (T_CMB_ary / T_s_ary) * (omega_b_h / 0.0327) 
    #     * (phys.omega_m / 0.307)**(-0.5) * np.sqrt((1. + z_ary) / 10.)
    # )
    # tau_ary = (
    #     3 * phys.c * (21.1 ** 2) * (2 * np.pi * phys.hbar) * 2.85e-15 * phys.nH * (1. + z_ary)**3 
    #     / (32 * np.pi * phys.kB * T_s_ary * (1. + z_ary) * phys.hubble(1. + z_ary) / (1. + z_ary))
    # )

    # Use full expression. 
    A_21     = 2.85e-15 # decay constant of excited state in sec^-1
    omega_10 = 0.0681462 # hyperfine splitting in K
    k_10     = omega_10 * phys.kB / (phys.hbar * phys.c)   # hyperfine splitting in cm^-1
    
    xi_ary = omega_10 / T_s_ary     

    tau_ary = 3 * np.pi**2 * A_21 * phys.nH * (1. + z_ary)**3 / phys.hubble(1. + z_ary) / k_10**3 * (
        (1. - np.exp(-xi_ary)) / (1. + 3. * np.exp(-xi_ary))
    )

    xi_corr_fac_ary = xi_ary * np.exp(-xi_ary) / (1. - np.exp(-xi_ary))

    return (T_s_ary * xi_corr_fac_ary - T_CMB_ary) * (1. - np.exp(-tau_ary)) / (1. + z_ary)


