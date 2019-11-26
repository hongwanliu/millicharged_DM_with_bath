""" Integrals over form factors.
"""

import json

import numpy as np

from scipy.special import expi
from scipy.interpolate import interp2d

import physics as phys

dir_str = '/home/hongwanl/millicharged_DM_with_bath/data/'
J_over_r_arr      = np.array(json.load(open(dir_str+'JOverrTable.txt')))
J_diff_over_r_arr = np.array(json.load(open(dir_str+'JDiffOverrTable.txt')))
J_Vrel_over_r_arr = np.array(json.load(open(dir_str+'JVrelOverrTable.txt')))
K_over_r_arr      = np.array(json.load(open(dir_str+'KOverrTable.txt')))
K_diff_over_r_arr = np.array(json.load(open(dir_str+'KDiffOverrTable.txt')))
K_Vrel_over_r_arr = np.array(json.load(open(dir_str+'KVrelOverrTable.txt')))

log10eps_table = np.log10(J_over_r_arr[:,0,0])
log10xi_table  = np.log10(K_over_r_arr[:,0,0])
log10r_table   = np.log10(J_over_r_arr[0,:,1])


J_over_r_interp      = interp2d(
    log10eps_table, log10r_table, np.transpose(J_over_r_arr[:,:,2]),
    bounds_error=False
)
J_diff_over_r_interp = interp2d(
    log10eps_table, log10r_table, np.transpose(J_diff_over_r_arr[:,:,2]), 
    bounds_error=False
)
J_Vrel_over_r_interp = interp2d(
    log10eps_table, log10r_table, np.transpose(J_Vrel_over_r_arr[:,:,2]),
    bounds_error=False
)

K_over_r_interp      = interp2d(
    log10xi_table, log10r_table, np.transpose(K_over_r_arr[:,:,2]), 
    bounds_error=False
)
K_diff_over_r_interp = interp2d(
    log10xi_table, log10r_table, np.transpose(K_diff_over_r_arr[:,:,2]), 
    bounds_error=False
)
K_Vrel_over_r_interp = interp2d(
    log10xi_table, log10r_table, np.transpose(K_Vrel_over_r_arr[:,:,2]), 
    bounds_error=False
)

def J_over_r(eps, r):
    if r != 0:
        if np.log10(r) >= log10r_table[0]:
            return J_over_r_interp(np.log10(eps), np.log10(r))
        else:
            return J_over_r_interp(np.log10(eps), log10r_table[0])
    else:
        if eps < 30:
            exp_times_expi = np.exp(eps**2/2)*expi(-eps**2/2)
            return (1/np.sqrt(2*np.pi))*(-2 - exp_times_expi *(2 + eps**2))
        else:
            # Expand entire expression in parenthesis. 
            eps_term = (
                8/eps**4 - 64/eps**6 + 576/eps**8
                -6144/eps**10 + 76800/eps**12
                -1105920/eps**14 + 18063360/eps**16
            )
            return (1/np.sqrt(2*np.pi))*eps_term


def J_diff_over_r(eps, r):
    if r != 0:
        if np.log10(r) >= log10r_table[0]:
            return J_diff_over_r_interp(np.log10(eps), np.log10(r))
        else:
            return J_diff_over_r_interp(np.log10(eps), log10r_table[0])
    else:
        # To avoid overflow problems with np.exp(eps**2/2). 
        if eps < 30:
            exp_times_expi = np.exp(eps**2/2)*expi(-eps**2/2)
            return 1. + 0.5*exp_times_expi*(2 + eps**2)
        else:
            # Expand entire expression. 
            return (
                -4/eps**4 + 32/eps**6 - 288/eps**8
                +3072/eps**10 - 38400/eps**12 + 552960/eps**14
                -9031680/eps**16
            )
        
def J_Vrel_over_r(eps, r):
    if r!= 0:
        if np.log10(r) >= log10r_table[0]:
            return J_Vrel_over_r_interp(np.log10(eps), np.log10(r))
        else:
            return 1/(30*np.sqrt(2*np.pi))*r**2*(
                -20 + 2*(5+eps**2)*r**2 
                + (-10*(2 + eps**2) + (6 + 7*eps**2 + eps**4)*r**2) 
                    * np.exp(eps**2/2) * expi(-eps**2/2)
            )
    else:
        return 0.

def K_over_r(xi, r):
    if r != 0:
        if np.log10(r) >= log10r_table[0]:
            return K_over_r_interp(np.log10(xi), np.log10(r))
        else:
            if xi < 10:
                return 1/(144*np.sqrt(2*np.pi))*(
                    (2*(-6*(-4 - 8*xi**2 + xi**4) + r**2*(44 - 48*xi**2 - xi**4 + xi**6)))
                    + (
                        - 6 * (48 - 24*xi**2 - 6*xi**4 + xi**6)
                        + r**2 * (48 - 24*xi**2 - 54*xi**4 + xi**6 + xi**8)
                    ) * np.exp(xi**2/2) * expi(-xi**2/2)
                )
            else:
                return (
                    (8 * np.sqrt(2/np.pi) * (2+r**2))/xi**4 
                    + (-55296 - 46080*r**2)/ (144 * np.sqrt(2*np.pi)  * xi**6)
                    + (691200 + 806400*r**2)/(144 * np.sqrt(2*np.pi)  * xi**8)
                    + (-9732096 - 14598144*r**2)/(144 * np.sqrt(2*np.pi) * xi**10)
                )
    else:
        # To avoid overflow problems with np.exp(xi**2/2). 
        if xi < 30:
            exp_times_expi = np.exp(xi**2/2)*expi(-xi**2/2)
            return 1/(24*np.sqrt(2*np.pi))*(
                8 + 16*xi**2 - 2*xi**4 - exp_times_expi*(
                    48 - 24*xi**2 - 6*xi**4 + xi**6
                )
            )
        else:
            # Need to expand the full expression in the parenthesis
            # to obtain the correct xi expansion.

            xi_func = (
                768/xi**4 - 9216/xi**6 + 115200/xi**8
                - 1622016/xi**10 + 25804800/xi**12
                - 460062720/xi**14 + 9103933440/xi**16
            )

            return 1/(24*np.sqrt(2*np.pi))*xi_func
           

def K_diff_over_r(xi, r):
    if r > 10:
        0
#         print('your r value is: ', r)
#         raise TypeError('The interpolation table is not trustworthy for r > 10.')
    if r != 0:
        if np.log10(r) >= log10r_table[0]:
            return K_diff_over_r_interp(np.log10(xi), np.log10(r))
        else:
            if xi < 10:
                return 1/288*(
                    12*(-4 - 8*xi**2 + xi**4) - 2*r**2*(36 - 64*xi**2 + xi**4 + xi**6)
                    - (
                        -6*(48 - 24*xi**2 - 6*xi**4 + xi**6) 
                        + r**2*(144 - 72*xi**2 - 66*xi**4 + 3*xi**6 + xi**8)
                    ) * np.exp(xi**2/2) * expi(-xi**2/2)
                )
            else:
                return (
                    -8*(6 + r**2)/(3*xi**4)
                    + 96*(2 + r**2)/xi**6
                    - 400*(6 + 5*r**2)/xi**8
                    + 5632*(6 + 7*r**2)/xi**10
                )
            return K_diff_over_r_interp(np.log10(xi), log10r_table[0])
    else:
        # To avoid overflow problems with np.exp(xi**2/2). 
        if xi < 30:
            exp_times_expi = np.exp(xi**2/2)*expi(-xi**2/2)
            return 1/48*(
                2*(-4 - 8*xi**2 + xi**4)
                + exp_times_expi*(48 - 24*xi**2 - 6*xi**4 + xi**6)
            )
        else:
            # Need to expand the full expression in the parenthesis
            # to obtain the correct xi expansion.

            xi_func = -(
                768/xi**4 - 9216/xi**6 + 115200/xi**8
                - 1622016/xi**10 + 25804800/xi**12
                - 460062720/xi**14 + 9103933440/xi**16
            )
        return 1/48*xi_func
    
def K_Vrel_over_r(xi, r):
    if r!= 0:
        if np.log10(r) >= log10r_table[0]:
            # print('large r!')
            # print(K_Vrel_over_r_interp(np.log10(xi), np.log10(r)))
            return K_Vrel_over_r_interp(np.log10(xi), np.log10(r))
        else:
            if xi < 10:
                # print('small r!')
                # print(1/(720*np.sqrt(2*np.pi))*r**2*(
                #     2*(40 + 36*r**2 + xi**6*r**2 + xi**2*(80-64*r**2) + xi**4*(-10 + r**2))
                #     + (xi**8*r**2 + xi**2*(240 - 72*r**2) + xi**4*(60 - 66*r**2) + 48*(-10 + 3*r**2) + xi**6*(-10 + 3*r**2))
                #         * np.exp(xi**2/2) * expi(-xi**2/2)
                # ))
                return 1/(720*np.sqrt(2*np.pi))*r**2*(
                    2*(40 + 36*r**2 + xi**6*r**2 + xi**2*(80-64*r**2) + xi**4*(-10 + r**2))
                    + (xi**8*r**2 + xi**2*(240 - 72*r**2) + xi**4*(60 - 66*r**2) + 48*(-10 + 3*r**2) + xi**6*(-10 + 3*r**2))
                        * np.exp(xi**2/2) * expi(-xi**2/2)
                )
            else:
                return -8*np.sqrt(2/np.pi)*r**2/(15*xi**10)*(
                    9*r**2*(16 - 10*xi**2 - 6*xi**4 + xi**6) 
                    - 10*(48 - 30*xi**2 - 2*xi**4 + xi**6)
                )
    else:
        return 0.

def sigma_bar(m_chi, Q):
    # m_chi in eV, returns cross section in eV^-2.
    mu_e  = m_chi*phys.me/(m_chi + phys.me)
    return 16*np.pi * mu_e**2 * Q**2 * phys.alpha**2 / (
        (phys.alpha*phys.me)**4
    )



def m_phot(xe, rs, T_m):
    
    # T in eV, returns mass in eV. 
    ele_squared = 4*np.pi*phys.alpha
    ne_eV_cubed = xe*phys.nH*rs**3*(phys.hbar*phys.c)**3

    return np.sqrt(4*np.pi*ne_eV_cubed*ele_squared/T_m)

def m_dark_phot(f, alpha, beta, m_chi, Q_d, rs, T_zeta):
    # Consistent with SM m_phot, taking only the temperature
    # of the dominant component. 

    if alpha == 0:
        # No dark bath. 
        return 0

    ele_squared = 4*np.pi*phys.alpha*Q_d**2
    n_eV_cubed  = alpha*beta*phys.rho_DM*rs**3/m_chi 
    n_eV_cubed  *= (phys.hbar*phys.c)**3

    return np.sqrt(4*np.pi*n_eV_cubed*ele_squared/T_zeta)

def I_v(m_chi, Q, T_chi, T_m, V_rel, xe, rs, species):
    
    if species == 'e':
        m = phys.me
        born_fac = 2.
    elif species == 'p':
        m = phys.mp
        born_fac = 2.
    elif species == 'H':
        m = phys.mp
        born_fac = 1.
    elif species == 'He':
        m = phys.mHe
        born_fac = 1.

    # Factor of 2 to convert from Born to classical. 

    xsec = born_fac*sigma_bar(m_chi, Q)

    
    u    = np.sqrt(T_chi/m_chi + T_m/m)
    r    = V_rel/u
    mu   = m_chi*m/(m_chi + m)
    mu_e = m_chi*phys.me/(m_chi + phys.me)
    
    prefac = xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)
    
    # Converts to cm^2
    prefac *= (phys.hbar*phys.c)**2
    
    if species == 'e' or species == 'p':
        m_phi = m_phot(xe, rs, T_m)
        # Born to classical correction
        m_eff = np.sqrt(4. * mu * m_phi * Q * phys.alpha / np.exp(1.))
        eps = m_eff/(2 * mu * u)
#         print('rs: ', rs, 'eps: ', eps, 'r: ', r)
        return prefac * J_over_r(eps, r)
    elif species == 'H':
        bohr_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
        xi = 1/(bohr_rad * mu * u)
        return prefac * K_over_r(xi, r)
    elif species == 'He':
        boh_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
        xi = 1.69/2/(bohr_rad * mu * u)
        return 4 * prefac * K_over_r(xi, r)
    else:
        raise TypeError('invalid species.')

def I_v_DM(
    f, alpha, beta, m_chi, Q_d, T_chi, T_zeta, V_chi_zeta, rs
):

    if alpha == 0:
        return 0

    m_zeta = m_chi/beta

    # Q_d is sqrt (Q_m Q_c). cross section is proportional to Q_d^4. 
    # Factor of 2 for classical correction.
    xsec = 2.*sigma_bar(m_chi, Q_d**2)

    u    = np.sqrt(T_chi/m_chi + T_zeta/m_zeta)
    r    = V_chi_zeta/u
    mu   = m_chi*m_zeta/(m_chi + m_zeta)
    mu_e = m_chi*phys.me/(m_chi + phys.me)

    prefac = xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)
    
    # Converts to cm^2
    prefac *= (phys.hbar*phys.c)**2

    m_phi = m_dark_phot(f, alpha, beta, m_chi, Q_d, rs, T_zeta)
    m_eff = np.sqrt(4. * mu * m_phi * Q_d**2 * phys.alpha/np.exp(1.))
    eps = m_eff/(2 * mu * u)

    return prefac * J_over_r(eps, r)

    
def I_V_minus_I_v(m_chi, Q, T_chi, T_m, V_rel, xe, rs, species):
    
    
    if species == 'e':
        m = phys.me
        born_fac = 2.
    elif species == 'p':
        m = phys.mp
        born_fac = 2.
    elif species == 'H':
        m = phys.mp
        born_fac = 1.
    elif species == 'He':
        m = phys.mHe
        born_fac = 1.

    xsec = born_fac*sigma_bar(m_chi, Q)
    
    u    = np.sqrt(T_chi/m_chi + T_m/m)
    r    = V_rel/u
    mu   = m_chi*m/(m_chi + m)
    mu_e = m_chi*phys.me/(m_chi + phys.me)
    
    prefac = (
        np.sqrt(2/np.pi)*xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)
    )
    
    # Converts to cm^2
    prefac *= (phys.hbar*phys.c)**2
    
    if species == 'e' or species == 'p':
        m_phi = m_phot(xe, rs, T_m)
        # Born to classical correction
        m_eff = np.sqrt(4 * mu * m_phi * Q * phys.alpha / np.exp(1.))
        eps = m_eff/(2 * mu * u)
#         print('eps: ', eps, 'r: ', r)
        return prefac * J_diff_over_r(eps, r)
    elif species == 'H':
        bohr_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
        xi = 1/(bohr_rad * mu * u)
        return prefac * K_diff_over_r(xi, r)
    elif species == 'He':
        bohr_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
        xi = 1.69/2/(bohr_rad * mu * u)
        return 4 * prefac * K_diff_over_r(xi, r)
    else:
        raise TypeError('invalid species.')

def I_V_minus_I_v_DM(
    f, alpha, beta, m_chi, Q_d, T_chi, T_zeta, V_chi_zeta, rs
):

    if alpha == 0:
        # No dark bath. 
        return 0

    m_zeta = m_chi/beta

    # Q_d is sqrt (Q_m Q_c). cross section is proportional to Q_d^4. 
    # Factor of 2 for classical correction.
    xsec = 2.*sigma_bar(m_chi, Q_d**2)

    u    = np.sqrt(T_chi/m_chi + T_zeta/m_zeta)
    r    = V_chi_zeta/u
    mu   = m_chi*m_zeta/(m_chi + m_zeta)
    mu_e = m_chi*phys.me/(m_chi + phys.me)

    prefac = (
        np.sqrt(2/np.pi)*xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)
    )
    
    # Converts to cm^2
    prefac *= (phys.hbar*phys.c)**2 

    m_phi = m_dark_phot(f, alpha, beta, m_chi, Q_d, rs, T_zeta)
    m_eff = np.sqrt(4. * mu * m_phi * Q_d**2 * phys.alpha/np.exp(1.))
    eps = m_eff/(2 * mu * u)

    return prefac * J_diff_over_r(eps, r)
        
def I_cap_V(m_chi, Q, T_chi, T_m, V_rel, xe, rs, species):
    
    if V_rel != 0:
        
        if species == 'e':
            m = phys.me
            born_fac = 2.
        elif species == 'p':
            m = phys.mp
            born_fac = 2.
        elif species == 'H':
            m = phys.mp
            born_fac = 1.
        elif species == 'He':
            m = phys.mHe
            born_fac = 1.

        xsec = born_fac*sigma_bar(m_chi, Q)

        u    = np.sqrt(T_chi/m_chi + T_m/m)
        r    = V_rel/u
        mu   = m_chi*m/(m_chi + m)
        mu_e = m_chi*phys.me/(m_chi + phys.me)

        prefac = xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)

        # Converts to cm^2
        prefac *= (phys.hbar*phys.c)**2

        if species == 'e' or species == 'p':
            m_phi = m_phot(xe, rs, T_m)
            # Born to classical correction
            m_eff = np.sqrt(4 * mu * m_phi * Q * phys.alpha / np.exp(1.))
            eps = m_eff/(2 * mu * u)
            return prefac * J_Vrel_over_r(eps, r)
        elif species == 'H':
            bohr_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
            xi = 1/(bohr_rad * mu * u)
            # print('xi, r: ', xi, r)
            return prefac * K_Vrel_over_r(xi, r)
        elif species == 'He':
            bohr_rad = phys.bohr_rad/(phys.hbar*phys.c) # in eV^-1
            xi = 1.69/2/(bohr_rad * mu * u)
            return 4 * prefac * K_Vrel_over_r(xi, r)
        else:
            raise TypeError('invalid species.')
    
    else:
        return 0

def I_cap_V_DM(
    f, alpha, beta, m_chi, Q_d, T_chi, T_zeta, V_chi_zeta, rs
):
    
    if alpha != 0 and V_chi_zeta != 0:
        # alpha = 0 means no dark bath.

        m_zeta = m_chi/beta

        xsec = 2.*sigma_bar(m_chi, Q_d**2)

        u    = np.sqrt(T_chi/m_chi + T_zeta/m_zeta)
        r    = V_chi_zeta/u
        mu   = m_chi*m_zeta/(m_chi + m_zeta)
        mu_e = m_chi*phys.me/(m_chi + phys.me)

        prefac = xsec*(phys.alpha*phys.me)**4/(8*u * mu**2 * mu_e**2)

        # Converts to cm^2
        prefac *= (phys.hbar*phys.c)**2

        m_phi = m_dark_phot(f, alpha, beta, m_chi, Q_d, rs, T_zeta)
        m_eff = np.sqrt(4. * mu * m_phi * Q_d**2 * phys.alpha/np.exp(1.))
        eps = m_eff/(2 * mu * u)  

        return prefac * J_Vrel_over_r(eps, r)

    else:
        return 0 


