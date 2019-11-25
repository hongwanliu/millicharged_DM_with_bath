""" Differential equations to solve for multicomponent DM, reformulated.
"""

import numpy as np

import physics as phys
import integrals as integ

def D_bm(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C,
    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
    f
):

    # d|V_bm|/dt

    alpha = 1. - f
    beta  = m_m/m_C

    T_b = phys.TCMB(rs) - Delta_CMB_b
    T_m = T_b - Delta_b_m
    T_C = T_m - Delta_m_C

    rho_m_0 = f * phys.rho_DM
    mu_mC   = m_m * m_C/(m_m + m_C)
    n_C     = (1-f) * phys.rho_DM * rs**3 / m_C

    species_list = ['e', 'p']
    mu_list      = [
        m_m * phys.me / (m_m + phys.me), 
        m_m * phys.mp / (m_m + phys.mp)
    ]
    n_list       = [
        xe * phys.nH * rs**3, 
        xe * phys.nH * rs**3
    ]

    if neutral_H:
        species_list.append('H')
        mu_list.append(m_m * phys.mp/(m_m + phys.mp))
        n_list.append((1-xe) * phys.nH * rs**3)

    if neutral_He:
        species_list.append('He')
        mu_list.append(m_m * phys.mHe/(m_m + phys.mHe))
        n_list.append(phys.nHe * rs**3)

    summand = 0
    for species,mu,n in zip(species_list, mu_list, n_list):

        summand += mu * n / m_m * integ.I_cap_V(
            m_m, Q, T_m, T_b, V_bm, xe, rs, species
        )

    if V_bm == 0:
        term_1 = 0
    else:
        term_1 = -(1 + rho_m_0/phys.rho_baryon) * summand / V_bm

    if Q_d != 0 and V_mC != 0:
        term_2 = mu_mC * n_C / m_m * integ.I_cap_V_DM(
            f, alpha, beta, m_m, Q_d, T_m, T_C, V_mC, rs
        ) / V_mC
    else:
        term_2 = 0

    # Multiply by c to convert from cm^-1 to s^-1
    return (term_1 + term_2) * phys.c

def D_mC(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C,
    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
    f
):

    # d|V_mC|/dt

    alpha = 1. - f
    beta  = m_m/m_C

    T_b = phys.TCMB(rs) - Delta_CMB_b
    T_m = T_b - Delta_b_m
    T_C = T_m - Delta_m_C

    mu_mC   = m_m * m_C/(m_m + m_C)
    n_C     = (1.-f) * phys.rho_DM * rs**3 / m_C
    rho_C   = (1.-f) * phys.rho_DM * rs**3
    rho_m   =    f  * phys.rho_DM * rs**3

    species_list = ['e', 'p']
    mu_list      = [
        m_m * phys.me / (m_m + phys.me), 
        m_m * phys.mp / (m_m + phys.mp)
    ]
    n_list       = [
        xe * phys.nH * rs**3, 
        xe * phys.nH * rs**3
    ]

    if neutral_H:
        species_list.append('H')
        mu_list.append(m_m * phys.mp/(m_m + phys.mp))
        n_list.append((1-xe) * phys.nH * rs**3)

    if neutral_He:
        species_list.append('He')
        mu_list.append(m_m * phys.mHe/(m_m + phys.mHe))
        n_list.append(phys.nHe * rs**3)

    summand = 0
    for species, mu, n in zip(species_list, mu_list, n_list):

        summand += mu * n / m_m * integ.I_cap_V(
            m_m, Q, T_m, T_b, V_bm, xe, rs, species
        )

    if V_bm == 0:
        term_1 = 0.
    else:
        term_1 = summand/V_bm

    if Q_d == 0 or V_mC == 0:
        term_2 = 0. 
    else:
        term_2 = -(1 + rho_m/rho_C) * mu_mC * n_C / m_m * (
            integ.I_cap_V_DM(
                f, alpha, beta, m_m, Q_d, T_m, T_C, V_mC, rs
            ) / V_mC
        )

    return (term_1 + term_2) * phys.c

def Q_i_dot(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C, 
    V_bm, V_mC, xe, rs, species, neutral_H, neutral_He,
    f
):

    if species == 'e':
        m_i  = phys.me
    elif species == 'p':
        m_i  = phys.mp
    elif species == 'H':
        m_i = phys.mp
    elif species == 'He':
        m_i = phys.mHe
    else:
        raise TypeError('invalid species.')

    T_b = phys.TCMB(rs) - Delta_CMB_b
    T_m = T_b - Delta_b_m
    T_C = T_m - Delta_m_C

    mu_i  = (m_m * m_i) / (m_m + m_i)
    u_m   = np.sqrt(T_m / m_m)
    u_i   = np.sqrt(T_b / m_i)

    term_1_prefac = Delta_b_m / ((u_m**2 + u_i**2) * m_i * m_m)
    term_1 = term_1_prefac * integ.I_V_minus_I_v(
        m_m, Q, T_m, T_b, V_bm, xe, rs, species
    )

    term_2 = (1/m_i) * integ.I_cap_V(
        m_m, Q, T_m, T_b, V_bm, xe, rs, species
    )

    n_m = (f * phys.rho_DM * rs ** 3)/m_m 

    return n_m * mu_i ** 2 * (term_1 + term_2) * phys.c

def n_Q_i_dot_sum(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C,
    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
    f, neutrals_only=False
): 
   
    n_list = [
        xe*phys.nH*rs**3,
        xe*phys.nH*rs**3,
        (1-xe)*phys.nH*rs**3,
        phys.nHe*rs**3
    ]
    
    if not neutrals_only:
        sum_e_p = (
            n_list[0] * Q_i_dot(
                m_m, m_C, Q, Q_d, 
                Delta_CMB_b, Delta_b_m, Delta_m_C,
                V_bm, V_mC, xe, rs, 'e', neutral_H, neutral_He, 
                f
            )
            + n_list[1] * Q_i_dot(
                m_m, m_C, Q, Q_d, 
                Delta_CMB_b, Delta_b_m, Delta_m_C,
                V_bm, V_mC, xe, rs, 'p', neutral_H, neutral_He, 
                f
            )
        )
    else:
        sum_e_p = 0.

    sum_to_return = sum_e_p

    if neutral_H: 

        sum_to_return += n_list[2] * Q_i_dot(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, 'H', neutral_H, neutral_He, 
            f
        )

    if neutral_He: 

        sum_to_return += n_list[3] * Q_i_dot(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, 'He', neutral_H, neutral_He, 
            f
        )

    return sum_to_return

def Q_C_dot(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C,
    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
    f
):

    alpha = 1. - f
    beta  = m_m/m_C

    if Q_d == 0:

        return 0. 

    T_b = phys.TCMB(rs) - Delta_CMB_b
    T_m = T_b - Delta_b_m
    T_C = T_m - Delta_m_C

    mu_mC = (m_m * m_C) / (m_m + m_C)

    u_m   = np.sqrt(T_m / m_m)
    u_C   = np.sqrt(T_C / m_C)

    term_1_prefac = Delta_m_C / ((u_m ** 2 + u_C ** 2) * m_C * m_m)
    term_1 = -term_1_prefac * integ.I_V_minus_I_v_DM(
        f, alpha, beta, m_m, Q_d, T_m, T_C, V_mC, rs
    )

    term_2 = (1 / m_C) * integ.I_cap_V_DM(
        f, alpha, beta, m_m, Q_d, T_m, T_C, V_mC, rs
    )

    n_m = (f * phys.rho_DM * rs ** 3)/m_m 

    return n_m * mu_mC ** 2 * (term_1 + term_2) * phys.c

def Q_m_dot(
    m_m, m_C, Q, Q_d, 
    Delta_CMB_b, Delta_b_m, Delta_m_C,
    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
    f, neutrals_only=False
):
    
    alpha = 1. - f
    beta  = m_m/m_C

    rho_m    = f * phys.rho_DM * rs ** 3
    n_m      = rho_m / m_m
    mu_mC    = (m_m * m_C) / (m_m + m_C)
    rho_C    = (1-f) * phys.rho_DM * rs ** 3
    n_C      = rho_C / m_C

    T_b = phys.TCMB(rs) - Delta_CMB_b
    T_m = T_b - Delta_b_m
    T_C = T_m - Delta_m_C

    species_list = ['e', 'p']
    mu_list      = [
        m_m * phys.me / (m_m + phys.me), 
        m_m * phys.mp / (m_m + phys.mp)
    ]
    n_list       = [
        xe * phys.nH * rs**3, 
        xe * phys.nH * rs**3
    ]

    if neutral_H:
        species_list.append('H')
        mu_list.append(m_m * phys.mp/(m_m + phys.mp))
        n_list.append((1-xe) * phys.nH * rs**3)

    if neutral_He:
        species_list.append('He')
        mu_list.append(m_m * phys.mHe/(m_m + phys.mHe))
        n_list.append(phys.nHe * rs**3)

    summand = 0
    for species, mu, n in zip(species_list, mu_list, n_list):

        if neutrals_only:

            if not neutral_H and not neutral_He:

                raise ValueError('neutral_H or neutral_He must be specified to use neutrals_only')

            if species != 'e' and species != 'p':

                summand += mu * n * integ.I_cap_V(
                    m_m, Q, T_m, T_b, V_bm, xe, rs, species
                )

        else:

            summand += mu * n * integ.I_cap_V(
                m_m, Q, T_m, T_b, V_bm, xe, rs, species
            )

    # summand is in eV/cm
    term_1 = summand * phys.c

    if Q_d != 0:
        term_2 = n_C * mu_mC * integ.I_cap_V_DM(
            f, alpha, beta, m_m, Q_d, T_m, T_C, V_mC, rs
        ) * phys.c

    else:

        term_2 = 0.

    term_3 = - n_Q_i_dot_sum(
        m_m, m_C, Q, Q_d, 
        Delta_CMB_b, Delta_b_m, Delta_m_C,
        V_bm, V_mC, xe, rs, neutral_H, neutral_He,
        f, neutrals_only=neutrals_only
    ) / n_m

    if Q_d != 0:

        # Q_C_dot doesn't actually depend on neutral_H and neutral_He.

        term_4 = -n_C * Q_C_dot(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, neutral_H, neutral_He,
            f
        ) / n_m
    
    else:

        term_4 = 0.

    return term_1 + term_2 + term_3 + term_4

#####################################################
#
# Actual set of ODEs to integrate.
#
#####################################################

def compton_cooling_rate(xe, Delta_CMB_b, rs):

    return (
        xe / (1 + xe + phys.chi) * Delta_CMB_b
        * 32 * phys.thomson_xsec * phys.stefboltz
        * phys.TCMB(rs)**4 / (3 * phys.me)
    )

def time_fac(rs):
    # returns H(z) (1+z). 
    return phys.hubble(rs)*rs

def DM_baryon_ODE(
    rs, var, m_m, m_C, Q, Q_d, neutral_H, neutral_He, f, log_T_C, 
    zero_V_rel=False, neutrals_only=False
):

    def dDelta_CMB_b_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C


        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)

        rate = (
            phys.TCMB(1) - (
                2 * T_b / rs
                - (2/3)*(
                    n_Q_i_dot_sum(
                        m_m, m_C, Q, Q_d, 
                        Delta_CMB_b, Delta_b_m, Delta_m_C,
                        V_bm, V_mC, xe, rs, neutral_H, neutral_He,
                        f, neutrals_only=neutrals_only
                    )
                    /time_fac(rs)/(phys.nH*rs**3)/(1 + xe + phys.chi)
                )
                - compton_cooling_rate(xe, Delta_CMB_b, rs)/time_fac(rs)
            )
        ) 

        # if Delta_CMB_b < 1e-7 and rate > 0:

        #     return 0

        # else:

        return rate

    def dDelta_b_m_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C

        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)

        rate = (
            2 * Delta_b_m / rs 
            - (2/3) * (
                n_Q_i_dot_sum(
                    m_m, m_C, Q, Q_d, 
                    Delta_CMB_b, Delta_b_m, Delta_m_C,
                    V_bm, V_mC, xe, rs, neutral_H, neutral_He,
                    f, neutrals_only=neutrals_only
                )
                /time_fac(rs)/(phys.nH*rs**3)/(1 + xe + phys.chi)
            )
            - compton_cooling_rate(xe, Delta_CMB_b, rs)/time_fac(rs)
            + (2/3) * Q_m_dot(
                m_m, m_C, Q, Q_d, 
                Delta_CMB_b, Delta_b_m, Delta_m_C,
                V_bm, V_mC, xe, rs, neutral_H, neutral_He,
                f, neutrals_only=neutrals_only
            ) / time_fac(rs)
        )

        # if Delta_b_m < 1e-7 and rate > 0:

        #     return 0

        # else:

        return rate

    def dlog_Delta_m_C_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C

        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)

        rate = (
            2 * Delta_m_C / rs 
            - (2/3) * Q_m_dot(
                m_m, m_C, Q, Q_d, 
                Delta_CMB_b, Delta_b_m, Delta_m_C,
                V_bm, V_mC, xe, rs, neutral_H, neutral_He,
                f, neutrals_only=neutrals_only
            ) / time_fac(rs)
            + (2/3) * Q_C_dot(
                m_m, m_C, Q, Q_d, 
                Delta_CMB_b, Delta_b_m, Delta_m_C,
                V_bm, V_mC, xe, rs, neutral_H, neutral_He,
                f
            ) / time_fac(rs)
        ) / Delta_m_C


        if Delta_m_C < 1e-7 and rate > 0:

            return 0

        else:

            return rate


    def dlog_T_C_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var,
        log_V_bm, log_V_mC, xe
    ): 

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C

        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)

        rate = 2 / rs - (2/3) * Q_C_dot(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, neutral_H, neutral_He,
            f
        ) / time_fac(rs) / T_C


        return rate

    def dlog_V_bm_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C

        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)


        rate = 1./rs - D_bm(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, neutral_H, neutral_He,
            f
        ) / V_bm / time_fac(rs)

        if (V_bm < 1e-15 and rate > 0) or zero_V_rel or rs > 1010:

            return 0

        else:

            return rate

    def dlog_V_mC_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C

        V_bm = np.exp(log_V_bm)
        V_mC = np.exp(log_V_mC)

        rate = 1./rs - D_mC(
            m_m, m_C, Q, Q_d, 
            Delta_CMB_b, Delta_b_m, Delta_m_C,
            V_bm, V_mC, xe, rs, neutral_H, neutral_He,
            f
        ) / V_mC / time_fac(rs)

        if (V_mC < 1e-15 and rate > 0) or zero_V_rel or rs > 1010:

            return 0

        else:

            return rate


    def dxe_dz(
        rs, Delta_CMB_b, Delta_b_m, T_C_var, 
        log_V_bm, log_V_mC, xe
    ):

        T_b = phys.TCMB(rs) - Delta_CMB_b
        T_m = T_b - Delta_b_m

        if not log_T_C:
            Delta_m_C = np.exp(T_C_var)
            T_C = T_m - Delta_m_C
        else:
            T_C = np.exp(T_C_var)
            Delta_m_C = T_m - T_C
        
        if xe > 0.99:
            # Use the Saha solution here. 
            def RHS(rs):
                de_broglie_wavelength = (
                    phys.c * 2*np.pi*phys.hbar
                    / np.sqrt(2 * np.pi * phys.me * phys.TCMB(rs))
                )
                return (
                    (1/de_broglie_wavelength**3) / (phys.nH*rs**3) 
                    * np.exp(-phys.rydberg/phys.TCMB(rs))
                )
            
            return (
                xe**2/(2*xe + RHS(rs)) *(1/rs) 
                * (phys.rydberg/phys.TCMB(rs) - 3/2)
            )
            
        else:
            
            T_r = phys.TCMB(rs)
            C = phys.peebles_C(xe, rs)
            alpha = phys.alpha_recomb(T_b)
            beta  = phys.beta_ion(T_r)

            return C*(
                alpha*xe**2*phys.nH*rs**3
                - 4*beta*(1-xe)*np.exp(-phys.lya_eng/T_r)
            )/time_fac(rs)


    Delta_CMB_b, Delta_b_m, T_C_var, log_V_bm, log_V_mC, xe = (
        var[0], var[1], var[2], var[3], var[4], var[5]
    )

    # print(
    #     rs, 
    #     Delta_CMB_b, 
    #     Delta_b_m, 
    #     T_C_var, 
    #     log_V_bm, 
    #     log_V_mC,
    #     xe
    # )
    # print(dlog_V_bm_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ))

    # print([
    #         dDelta_CMB_b_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ), 
    #         dDelta_b_m_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ), 
    #         dlog_T_C_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ), 
    #         dlog_V_bm_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ),
    #         dlog_V_mC_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         ),
    #         dxe_dz(
    #             rs, Delta_CMB_b, Delta_b_m, T_C_var, 
    #             log_V_bm, log_V_mC, xe
    #         )

    #     ]   )

    if not log_T_C:

        return [
            dDelta_CMB_b_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dDelta_b_m_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dlog_Delta_m_C_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dlog_V_bm_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ),
            dlog_V_mC_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ),
            dxe_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            )

        ]

    else:

        return [
            dDelta_CMB_b_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dDelta_b_m_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dlog_T_C_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ), 
            dlog_V_bm_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ),
            dlog_V_mC_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            ),
            dxe_dz(
                rs, Delta_CMB_b, Delta_b_m, T_C_var, 
                log_V_bm, log_V_mC, xe
            )

        ]        



