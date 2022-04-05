""" Calculating fluctuations from data. 
"""

import numpy as np
from scipy.interpolate import interp1d

import physics as phys 
import logfft 

import warnings

corrs=np.loadtxt(open('../correlations.dat'), delimiter="\t")

class Velocity_Fluctuations: 
    """Structure for velocity fluctuations. 

    Parameters
    ----------
    sigma3D : float
        3D velocity dispersion in km/s. 

    Attributes
    ----------
    cparint : function
        c_parallel velocity correlation (dimensionless). 
    cperpint : function
        c_perp velocity correlation (dimensionless). 
    sigma3D : float
        3D velocity dispersion in km/s. 
    sigma1D : float
        1D velocity dispersion in km/s. 
    
    """

    def __init__(self, sigma3D=29.): 

        # c_parallel, c_perp for velocity correlations
        self.cparint  = interp1d(corrs[:,0],corrs[:,1],kind=9)
        self.cperpint = interp1d(corrs[:,0],corrs[:,2],kind=9)
        
        # 3D velocity dispersion in km/s. 
        self.sigma3D  = sigma3D
        # 1D velocity dispersion in km/s. 
        self.sigma1D  = self.sigma3D / np.sqrt(3)

    ################# VELOCITY CORRELATORS ######################

    def get_xiv(self, x_ary): 
        """Velocity correlation function. 
        
        This is <v^i v_i>(x) or <V V>(x), in km^2/s^2. 

        Parameters
        ----------
        x_ary : ndarray
            Distances, in Mpc. 

        Returns
        -------
        ndarray

        """

        cpar_ary = np.zeros_like(x_ary)
        cpar_ary[x_ary <= 1e3] = self.cparint(x_ary[x_ary <= 1e3]) 

        cperp_ary = np.zeros_like(x_ary)
        cperp_ary[x_ary <= 1e3] = self.cperpint(x_ary[x_ary <= 1e3])

        return (cpar_ary + 2 * cperp_ary) * self.sigma1D**2

    def get_xiv2(self, x_ary): 
        """Velocity squared correlation function. 
        
        This is <v^2 v^2>(x), in km^4/s^4. 

        Parameters
        ----------
        x_ary : ndarray
            Distances, in Mpc. 

        Returns
        -------
        ndarray

        """
        cpar_ary = np.zeros_like(x_ary)
        cpar_ary[x_ary <= 1e3] = self.cparint(x_ary[x_ary <= 1e3]) 

        cperp_ary = np.zeros_like(x_ary)
        cperp_ary[x_ary <= 1e3] = self.cperpint(x_ary[x_ary <= 1e3])


        return 2 * (cpar_ary**2 + 2 * cperp_ary**2) * self.sigma1D**4

    def get_Delta2v(self, k_ary=None, x_ary=np.arange(0, 1000, 0.001), use_logfft=False, logrmin=-4, logrmax=4): 
        """Dimensionless power spectrum of V. 

        Result is in km^2/s^2.  

        Parameters
        ----------
        k_ary : ndarray, optional
            Wavenumbers. If x_ary is unspecified, should be in Mpc^-1.
        x_ary : ndarray, optional
            Distances. If unspecified, in Mpc. 
        use_logfft : bool, optional
            If True, uses log-FFT to evaluate. Otherwise, use direct numerical integration.
        logrmin : float, optional
            Minimum r value to use for log-FFT. 
        logrmax : float, optional
            Maximum r value to use for log-FFT. 
    
        Returns
        -------
        tuple of ndarray    
            (k_ary, result)
        """
        
        if use_logfft: 

            k_fft, P_fft = logfft.fftj0(self.get_xiv, logrmin, logrmax) 

            return (k_fft, P_fft * k_fft**3 / (2 * np.pi**2))

        else:
        
            xiv_ary = self.get_xiv(x_ary)

            return (k_ary, np.array([
                2. / np.pi * k**2 * np.trapz(
                    xiv_ary * x_ary * np.sin(k * x_ary), x_ary
                ) for k in k_ary
            ]))

    def get_Delta2v2(self, k_ary=None, x_ary=np.arange(0, 1000, 0.001), use_logfft=False, logrmin=-4, logrmax=4): 
        """Dimensionless power spectrum of v^2 
        
        Result is in km^4/s^4.  

        Parameters
        ----------
        k_ary : ndarray, optional
            Wavenumbers. If x_ary is unspecified, should be in Mpc^-1.
        x_ary : ndarray, optional
            Distances. If unspecified, in Mpc. 
        use_logfft : bool, optional
            If True, uses log-FFT to evaluate. Otherwise, use direct numerical integration.
        logrmin : float, optional
            Minimum r value to use for log-FFT. 
        logrmax : float, optional
            Maximum r value to use for log-FFT.
    
        Returns
        -------
        tuple of ndarray    
            (k_ary, result)

        """

        if use_logfft: 

            k_fft, P_fft = logfft.fftj0(self.get_xiv2, logrmin, logrmax)

            return (k_fft, P_fft * k_fft**3 / (2 * np.pi**2))

        else:

            xiv2_ary = self.get_xiv2(x_ary)

            return (k_ary, np.array([
                2. / np.pi * k**2 * np.trapz(
                    xiv2_ary * x_ary * np.sin(k * x_ary), x_ary
                ) for k in k_ary
            ]))

    ################# VELOCITY PDF QUANTITIES ######################

    def MB(self, v): 
        """Maxwell-Boltzmann PDF of 3D velocity. 
        
        Integrating over d^3 v gives 1. Result is in s^3/km^3.

        Parameters
        ----------
        v : ndarray
            Velocities, in km/s. 

        Returns
        -------
        ndarray


        """
        return np.sqrt(27 / (8. * np.pi**3)) / self.sigma3D**3 * (
            np.exp(-3. * v**2 / (2 * self.sigma3D**2)) 
        )

    def mean_f(self, f, v):
        """Mean of function f over Maxwell-Boltzmann PDF. 

        Parameters
        ----------
        f : function 
            Function of velocities. 
        v : ndarray
            Velocities, in km/s. 
        
        Returns
        -------
        float
        """

        return 4 * np.pi * np.trapz(self.MB(v) * v**2 * f(v), v, axis=0)

    def var_f(self, f, v): 
        """Variance of function f over Maxwell-Boltzmann PDF. 
        
        This is <f^2> - <f>^2. 

        Parameters
        ----------
        f : function 
            Function of velocities. 
        v : ndarray
            Velocities, in km/s. 
        
        Returns
        -------
        float
        """

        return 4 * np.pi * np.trapz(
            self.MB(v) * v**2 * (f(v) - self.mean_f(f, v))**2, v, axis=0
        )

    def bias_f(self, f, v): 
        """Bias parameter. 
        
        This parameter relates the v^2 (dimensionless) correlation function to the (dimensionless) correlation function of f(v^2) at long distances. It is given by (3/2) * (1 - <v^2 f> / (<f> vrms)), where vrms = <v^2> - <v>^2. 

        The result is dimensionless. 

        Parameters
        ----------
        f : function 
            Function of velocities. 
        v : ndarray
            Velocities, in km/s. 
        
        Returns
        -------
        float
        """

        # Better to numerically compute than to use exact value, 29 km/s. 
        Vrms_sq = self.mean_f(lambda x:x**2, v)

        if np.abs(Vrms_sq / self.sigma3D**2 - 1.) > 1e-2: 
            warnings.warn('Computed Vrms_sq is not self.sigma3D.')

        return 3 / 2 * (1. - (
            4. * np.pi * np.trapz(self.MB(v) * v**4 * f(v), v, axis=0)
            / Vrms_sq / self.mean_f(f, v)
        ))


class Fluctuations:

    def __init__(self, z_ary, v_ary, f_ary, sigma3D=29.): 
        """Structure for fluctuations. 

        Although written with T21 in mind, can be extended to anything. 

        Parameters
        ----------
        z_ary : ndarray
            Array of redshifts.
        v_ary : ndarray
            Baryon-CDM relative velocity abscissa in km/s.  
        f_ary : ndarray
            Function values, e.g. baryon temperature in K, dimensions v_ary x z_ary  
        sigma3D : float
            3D velocity dispersion in km/s. 

        Attributes
        ----------
        v_fluc : Velocity_Fluctuations
            Class containing velocity fluctuations information. 
        mean : float or ndarray 
            Mean value of f integrated over v_ary, dimensions z_ary. 
        var : float or ndarray
            Variance of f integrated over v_ary, dimensions z_ary. 
        b : float or ndarray
            Bias parameter of f, dimensions z_ary. 
        dT2 : float or ndarray
            <f'^2> ...
        """

        self.z_ary = z_ary
        self.v_ary = v_ary
        self.f_ary = f_ary 

        self.v_fluc = Velocity_Fluctuations(sigma3D=sigma3D)

        self.f_v    = interp1d(v_ary, f_ary, axis=0)

        self.mean   = self.v_fluc.mean_f(self.f_v, v_ary) 
        self.var    = self.v_fluc.var_f(self.f_v, v_ary) 
        self.b      = self.v_fluc.bias_f(self.f_v, v_ary)

        # RETURN TO THIS, INCORRECT FOR NOW.  
        self.dT2 = self.v_fluc.var_f(
            interp1d(v_ary[:-1], (np.diff(f_ary) / np.diff(v_ary))**2), 
            v_ary[:-1]
        )

    def xi_f_numerical(self, x_ary): 
        """Numerical calculation of the correlation function. 

        xi_f has units of f^2. 

        Parameters
        ----------
        x_ary : ndarray
            Distances in Mpc. 

        Returns
        -------
        """

        u_ary = self.v_ary / self.v_fluc.sigma1D
        u_ary = np.linspace(u_ary[0], u_ary[-1], 70)
        y_ary = np.linspace(-1, 1, 50)

        def R(y_ary=y_ary, x_ary=x_ary): 
            # y = cos theta
            term_1 = self.v_fluc.cperpint(x_ary)**2 
            term_2 = y_ary[:,None]**2 * (
                self.v_fluc.cparint(x_ary)**2 - self.v_fluc.cperpint(x_ary)**2
            )
            return np.sqrt(term_1 + term_2) 

        def dW(
            y_ary=y_ary, x_ary=x_ary, u1_ary=u_ary, u2_ary=u_ary
        ): 

            one_minus_R_sq = 1. - R(y_ary, x_ary)**2 
            u1_sq_plus_u2_sq = u1_ary[:,None]**2 + u2_ary[None,:]**2 

            term_1 = np.einsum(
                'ij,k,l->ijkl', 1./(R(y_ary, x_ary)*np.sqrt(one_minus_R_sq)), u1_ary, u2_ary
            ) / np.pi

            term_2a_exp = np.einsum(
                'ij,kl->ijkl', 1. / (2 * one_minus_R_sq), -u1_sq_plus_u2_sq
            )

            term_2b_exp = np.einsum(
                'ij,k,l->ijkl', 2 * R(y_ary, x_ary) / (2 * one_minus_R_sq),
                u1_ary, u2_ary
            )

            term_2 = 0.5*np.exp(term_2a_exp + term_2b_exp)

            term_3 = 1. - np.exp(-2*term_2b_exp)

            dW_zero_R = np.einsum(
                'ij,kl,k,l->ijkl',
                np.ones_like(one_minus_R_sq),
                np.exp(-u1_sq_plus_u2_sq/2) / np.pi, u1_ary**2, u2_ary**2
            )

            large_R_term = term_1 * term_2 * term_3 - dW_zero_R 

            small_R_term = np.einsum(
                'ij,kl,k,l->ijkl',
                R(y_ary, x_ary)**2, 
                np.exp(-u1_sq_plus_u2_sq / 2) / 6. / np.pi, 
                u1_ary**2 * (u1_ary**2 - 3.), u2_ary**2 * (u2_ary**2 - 3),
            )

            mask = np.zeros_like(large_R_term) + R(y_ary, x_ary)[:,:,None,None]
            
            large_R_mask = np.ones_like(mask)
            small_R_mask = np.ones_like(mask)
            
            large_R_mask[mask < 0.1] *= 0
            small_R_mask[mask >= 0.1] *= 0
            
            return large_R_term*large_R_mask + small_R_term*small_R_mask

        def dff(
            y_ary=y_ary, x_ary=x_ary, u1_ary=u_ary, u2_ary=u_ary
        ): 

            return np.einsum(
                'ijkl,k,l->ijkl',
                dW(y_ary, x_ary, u1_ary, u2_ary), 
                self.f_v(u1_ary * self.v_fluc.sigma1D),
                self.f_v(u2_ary * self.v_fluc.sigma1D)
            )

        return np.trapz(
            np.trapz(
                np.trapz(
                    dff(y_ary, x_ary, u_ary, u_ary), y_ary, axis=0
                ),
                u_ary, axis=-2
            ), 
            u_ary, axis=-1
        )

        





    






        
        






        



