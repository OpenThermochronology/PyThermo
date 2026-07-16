"""
crystal.py

Class crystal and mineral system specific sub-classes. Contains methods primarly focused on solving the diffusion equation, with production and alpha ejection, using a Crank-Nicolson approach.

"""
from .constants import (
    np, 
    U238_ppm_atom, 
    U235_ppm_atom, 
    Th_ppm_atom, 
    Sm_ppm_atom, 
    lambda_238, 
    lambda_238_yr, 
    lambda_235, 
    lambda_235_yr, 
    lambda_232, 
    lambda_232_yr, 
    lambda_147, 
    lambda_147_yr, 
    lambda_f, 
    gas_constant, 
    ap_density,
)
from .core_solvers import _CN_diffusion_core_wrapper, _mp_diffusion_core
from scipy.integrate import romb
import warnings

class crystal:
    def __init__(self):
        pass

    def alpha_ejection(
            self, radius, nodes, r_step, U_ppm, Th_ppm, Sm_ppm, mineral_type
        ):
        """ 
        Determines the 1D alpha ejection He profile from the center of the grain to the edge assuming a spherical geometry. Currently, the method assumes a uniform distribution of parent isotopes, as well as averaged, unchained ejection radii for these parent isotopes.

        Parameters 
        ----------

        radius: float
            Radius of a sphere with equivalent surfacea area to volume ratio as grain (in micrometers)

        nodes: int
            Number of nodes for 1D finite difference diffusion solver

        r_step: float
            Grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center
        
        U_ppm: float
            Concentration of uranium in zircon (in ppm)
        
        Th_ppm: float
            Concentration of thorium in zircon (in ppm)
        
        Sm_ppm: optional float
            Concentration of samarium in zircon (in ppm), default is 0 

        mineral_type: string
            Type of mineral, either 'apatite' or 'zircon', for selection of alpha stopping distances.
            

        Returns
        -------

        aej_U238, aej_U235, aej_Th, aej_Sm: arrays of floats
            Alpha ejection corrected U238, U235, Th, and Sm distribution for a 1D radial profile from center of the grain to edge, units in atoms/g

        corr_factors: dictionary of floats
            Various Ft correction factors and total values for parent isotopes in atoms/g
        
        """
        # unit conversion for U,Th,Sm inputs to atoms/g
        U238_atom = U_ppm * U238_ppm_atom
        U235_atom = U_ppm * U235_ppm_atom
        Th_atom = Th_ppm * Th_ppm_atom
        Sm_atom = Sm_ppm * Sm_ppm_atom

        # alpha stopping distances as reported in Ketcham et al. (2011) (https://doi.org/10.1016/j.gca.2011.10.011)
        # option for no alpha ejection   
        if(mineral_type == 'apatite'):         
            as_U238 = 18.81
            as_U235 = 21.80
            as_Th = 22.25
            as_Sm = 5.93
        elif(mineral_type == 'zircon'):
            as_U238 = 15.55
            as_U235 = 18.05
            as_Th = 18.43
            as_Sm = 4.76
        elif(mineral_type == 'none'):
            as_U238 = 0
            as_U235 = 0
            as_Th = 0
            as_Sm = 0

        # alpha ejection profile arrays for each isotope
        # compute radial positions
        r_pos = (np.arange(nodes) + 0.5) * r_step

        # create masks for each isotope to ensure no divide by zero or very near zero
        mask_U238 = r_pos >= (radius - as_U238)
        mask_U235 = r_pos >= (radius - as_U235)
        mask_Th = r_pos >= (radius - as_Th)
        mask_Sm = r_pos >= (radius - as_Sm)

        # safe division, only divides where mask is true
        # still encounter RuntimeWarning because of the - r_pos term, but these get discarded anyway with the mask so can ignore them
        # U238
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ejected_U238 = U238_atom * (
                0.5 + (
                    np.divide(
                        r_pos**2 + radius**2 - as_U238**2,
                        2 * r_pos,
                        where = mask_U238,
                        out = np.zeros(nodes)
                    ) - r_pos
                ) / (2 * as_U238)
            )
        aej_U238 = np.where(mask_U238, ejected_U238, U238_atom)

        # U235
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ejected_U235 = U235_atom * (
                0.5 + (
                    np.divide(
                        r_pos**2 + radius**2 - as_U235**2,
                        2 * r_pos,
                        where=mask_U235,
                        out=np.zeros(nodes)
                    ) - r_pos
                ) / (2 * as_U235)
            )
        aej_U235 = np.where(mask_U235, ejected_U235, U235_atom)

        # Th
        if Th_atom == 0:
            aej_Th = np.zeros(nodes)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ejected_Th = Th_atom * (
                    0.5 + (
                        np.divide(
                            r_pos**2 + radius**2 - as_Th**2,
                            2 * r_pos,
                            where=mask_Th,
                            out=np.zeros(nodes)
                        ) - r_pos
                    ) / (2 * as_Th)
                )
            aej_Th = np.where(mask_Th, ejected_Th, Th_atom)

        # Sm
        if Sm_atom == 0:
            aej_Sm = np.zeros(nodes)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ejected_Sm = Sm_atom * (
                    0.5 + (
                        np.divide(
                            r_pos**2 + radius**2 - as_Sm**2,
                            2 * r_pos,
                            where=mask_Sm,
                            out=np.zeros(nodes)
                        ) - r_pos
                    ) / (2 * as_Sm)
                )
            aej_Sm = np.where(mask_Sm, ejected_Sm, Sm_atom)
                
        # calculate alpha ejection correction factors for age equation
        outer = (np.arange(1, nodes + 1) * r_step)**3
        inner = (np.arange(nodes) * r_step)**3
        vol = outer - inner

        # weighted sums
        nft_U238 = np.sum(vol * aej_U238)
        dft_U238 = np.sum(vol * U238_atom)  
        nft_U235 = np.sum(vol * aej_U235)
        dft_U235 = np.sum(vol * U235_atom)

        total_U238 = U238_atom * np.sum(vol) * 8
        total_U235 = U235_atom * np.sum(vol) * 7

        if Th_atom == 0:
            dft_Th = 1.0
            total_Th = 0.0
            nft_Th = 0.0
        else:
            nft_Th   = np.sum(vol * aej_Th)
            dft_Th   = np.sum(vol * Th_atom)
            total_Th = Th_atom * np.sum(vol) * 6

        if Sm_atom == 0:
            dft_Sm  = 1.0
            total_Sm = 0.0
            nft_Sm  = 0.0
        else:
            nft_Sm   = np.sum(vol * aej_Sm)
            dft_Sm   = np.sum(vol * Sm_atom)
            total_Sm = Sm_atom * np.sum(vol)
                
        # calculate fts
        ft_U238 = nft_U238 / dft_U238
        ft_U235 = nft_U235 / dft_U235
        ft_Th = nft_Th / dft_Th
        ft_Sm = nft_Sm / dft_Sm

        # store in a dictionary
        corr_factors = {
            'total_U238':total_U238,
            'total_U235':total_U235,
            'total_Th':total_Th,
            'total_Sm':total_Sm,
            'ft_U238':ft_U238,
            'ft_U235':ft_U235,
            'ft_Th':ft_Th,
            'ft_Sm':ft_Sm,
        }
        
        return aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors

    def CN_diffusion(
            self, 
            nodes, 
            r_step, 
            tT_path, 
            diffs, 
            aej_U238, 
            aej_U235, 
            aej_Th, 
            aej_Sm, 
            init_He=None, 
            produce=True,
            divide=False,
            ):
        """ 
        Wrapper that calls the core function for solving the diffusion equation with production along a 1D radial profile using the Crank-Nicoloson finite difference scheme. Assumes no flux across the center, or inner boundary, node (Neumann boundary condition) and zero concentration along the outer boundary (Dirichlet boundary condition). Original solution and set-up for the algorithm comes from Ketcham (2005) (https://doi.org/10.2138/rmg.2005.58.11), specifically equation 21, and equations 22 and 26 for boundary conditions.
        
        Parameters
        ----------

        nodes: int
            Number of nodes for 1D finite difference diffusion solver

        r_step: float
            Grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center

        tT_path: 2D array of floats
            Time temperature history along which to calculate diffusion
        
        diffs: 1D array or list
            Calculated diffusivities at each time temperature step in tT_path (units of micrometer2/s), must be length of np.size(tT_path,0) - 1

        aej_U238: 1D array or list
            Alpha ejected 1D profile for 238U (in atoms/g), must be length of nodes

        aej_U235: 1D array or list
            Alpha ejected 1D profile for 235U (in atoms/g), must be length of nodes
        
        aej_Th: 1D array or list
            Alpha ejected 1D profile for 232Th (in atoms/g), must be length of nodes
        
        aej_Sm: 1D array or list
            Alpha ejected 1D profile for 147Sm (in atoms/g), must be length of nodes
        
        init_He: optional 1D array
            1D profile of atoms for lattice, must be length of nodes. Default is None. Must be in terms of radial position. 
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True'.

        divide: optional boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses. Default is 'False'.

        Returns
        -------

        He_profile: array of floats
            The 1D concentration profile of diffused He (units of atoms/g).

        """
        # damping parameters
        M = 100
        initial_damp = 20

        He_profile = _CN_diffusion_core_wrapper(
            nodes,
            r_step,
            tT_path,
            np.asarray(diffs, dtype=np.float64),
            np.asarray(aej_U238, dtype=np.float64),
            np.asarray(aej_U235, dtype=np.float64),
            np.asarray(aej_Th, dtype=np.float64),
            np.asarray(aej_Sm, dtype=np.float64),
            np.asarray(init_He, dtype=np.float64) if init_He is not None else np.zeros(nodes),
            1.0 if produce else 0.0,
            divide,
            M,
            initial_damp,
        )

        return He_profile
    
    def mp_diffusion(
            self, 
            nodes, 
            r_step, 
            tT_path, 
            diff_parameters, 
            tolerance, 
            aej_U238, 
            aej_U235, 
            aej_Th, 
            aej_Sm, 
            init_fast_He=None,
            init_lat_He=None,
            produce=True
            ):
        """ 
        Wrapper that calls the core function for solving diffusion in a crystal with multiple diffusion pathways, a fast path and a volume path, using a Crank-Nicoloson scheme. Based on the set-up of Lee and Aldama (1992) (https://doi.org/10.1016/0098-3004(92)90093-7) and their algorithm for solving equations 18a and 18b. Adapted by Guenthner et al. (in prep) (doi pending) for a tridiagonal setup, production terms, and Neumann and Dirichlet-type boundary conditions.

        Parameters
        ----------

        nodes: int
            Number of nodes for 1D finite difference diffusion solver

        r_step: float
            Grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center

        tT_path: 2D array of floats
            Time temperature history along which to calculate diffusion

        diff_parameters: dictionary of floats
            Fitted parameters for multi-path diffusion. Diffusivities at each time temperature step in tT_path (units of micrometer2/s), must be length of np.size(tT_path,0) - 1.

        tolerance: float
            Convergence criterion for iterative diffusion algorithm

        aej_U238: 1D array or list
            Alpha ejected 1D profile for 238U (in atoms/g), must be length of nodes

        aej_U235: 1D array or list
            Alpha ejected 1D profile for 235U (in atoms/g), must be length of nodes
        
        aej_Th: 1D array or list
            Alpha ejected 1D profile for 232Th (in atoms/g), must be length of nodes
        
        aej_Sm: 1D array or list
            Alpha ejected 1D profile for 147Sm (in atoms/g), must be length of nodes
        
        init_fast_He: optional 1D array
            1D profile of atoms for fast path, must be length of nodes. Default is None. Must be in terms of radial position (see mp_profile function).
        
        init_lat_He: optional 1D array
            1D profile of atoms for lattice, must be length of nodes. Default is None. Must be in terms of radial position (see mp_profile function). 
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True'.


        Returns
        -------

        bulk_He_profile: 1D array of floats
            The 1D concentration profile of diffused He in the bulk grain (units of atoms/g)
        
        fast_He_profile: 1D array of floats
            The 1D concentration profile of diffused He in the fast pathways (units of atoms/g)

        lat_He_profile: 1D array of floats
            The 1D concentration profile of diffused He in the lattice (units of atoms/g)

        """
        # unpack diff_parameters, D values have to be in units of micrometer**2/s
        D_sc = diff_parameters['D_sc']
        D_v = diff_parameters['D_v']
        kappa_1 = diff_parameters['kappa_1']
        kappa_2 = diff_parameters['kappa_2']
        f = diff_parameters['f']

        # damping parameters
        M = 100
        initial_damp = 20

        # set up initial arrays
        init_fast = np.asarray(init_fast_He, dtype=np.float64) if init_fast_He is not None else np.zeros(nodes)
        init_lat  = np.asarray(init_lat_He, dtype=np.float64) if init_lat_He is not None else np.zeros(nodes)

        bulk_He_profile, fast_He_profile, lat_He_profile, converged = _mp_diffusion_core(
                nodes,
                r_step,
                tT_path,
                D_sc,
                D_v,
                kappa_1,
                kappa_2,
                f,
                tolerance,
                np.asarray(aej_U238, dtype=np.float64),
                np.asarray(aej_U235, dtype=np.float64),
                np.asarray(aej_Th, dtype=np.float64),
                np.asarray(aej_Sm, dtype=np.float64),
                init_fast,
                init_lat,
                1.0 if produce else 0.0,
                M,
                initial_damp,
        )

        if not converged:
            warnings.warn(
                'Diffusion solver did not converge',
                RuntimeWarning,
                stacklevel=2
            )

        return bulk_He_profile, fast_He_profile, lat_He_profile
    
    def divide_tT(self, D, dt, temp, r_step, M, initial_damp):
        """
        Helper function for dividing up tT paths when Fourier number is large. Follows the approach of Britz et al. (2003) (https://www.doi.org.10.1016/S0097-8485(02)00075-X).

        Parameters
        ----------
        
        D: float
            Diffusivity at a given time-step. In microns^2/s.

        dt: float
            Length of time-step to be subdivided. In seconds.

        temp: float
            Temperature for the time-step. In Kelvin.

        r_step: float
            Node spacing. In microns.

        M: int
            Number of sub-divisions for the exponentially increasing damping function.

        initial_damp: int
            Number of sub-divions for the initial damping.


        Returns
        -------

        sub_tT: 2D array
            An array of time-temperature points.
        
        """
        fourier_set = 0.5
        dt_int = fourier_set * r_step**2 / D

        # ensure damping steps don't produce negative times when dt_int is large, short-circuit if dt_int is large
        if dt_int > initial_damp + M:
            n_steps = int(dt / dt_int) + 1
            sub_tT = np.zeros((n_steps, 2))
            sub_tT[:, 0] = dt - np.arange(n_steps) * dt_int
            sub_tT[:, 1] = temp
            return sub_tT

        # set up iterative Newton-Raphson solver for eq 24 from Britz et al. (2003)
        fourier_calc = D * dt / r_step**2
        beta_guess = 1.2
        f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
        f_beta_prime = 2*fourier_calc*beta_guess - M*beta_guess**(M-1)
        tolerance = 1e-3

        beta_diff = beta_guess

        # iterate to solve
        while abs(beta_diff) > tolerance:
            beta = beta_guess - f_beta / f_beta_prime
            beta_diff = beta_guess - beta
            beta_guess = beta
            f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
            f_beta_prime = 2 * fourier_calc * beta_guess - M * beta_guess**(M - 1)

        # add on the initial damping steps
        expansion_steps = initial_damp + M
        
        sub_tT = np.zeros((expansion_steps,2))
        sub_tT[0,0] = dt
        i_vals = np.arange(1, initial_damp)
        sub_tT[1:initial_damp, 0] = dt -  dt_int * (1 + i_vals)
        tau_1 = sub_tT[initial_damp - 1, 0] * (beta - 1) / (beta**M - 1)

        delta_t = 0
        i_vals = np.arange(M - 1)
        delta_t = np.cumsum(beta**i_vals) * tau_1
        sub_tT[initial_damp:initial_damp + M - 1,0] = sub_tT[initial_damp - 1, 0] - delta_t

        # add on temperatures
        sub_tT[:,1] = temp
                
        return sub_tT

    def He_date(self, total_He, corr_factors):
        """ 
        Calculate corrected date using a Newton-Raphson method. Precision is set to 1000 years.

        Parameters
        ----------
        
        total_He: float
            The total amount of He present in a given grain for a given time temperature history

        corr_factors: dictionary
            Dictionary containing various total isotope amounts, scaled by production and volume (base of 1/(4/3 * Pi)), and Ft corrections

        Returns
        -------

        corrected_date: float
            The alpha ejection corrected (U-Th)/He date (in Ma)

        """
        # get the guess to the 1000 year precision, date guess is in years
        tolerance = 1000.0
        date_guess = 100000000.0
        
        # get Fts from the corr_factors dictionary
        ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
        ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
        ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
        ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

        # set up Newton-Raphson equation
        f_date = (
            ft_U238 * (np.exp(lambda_238_yr * date_guess) - 1)
            + ft_U235 * (np.exp(lambda_235_yr * date_guess) - 1)
            + ft_Th * (np.exp(lambda_232_yr * date_guess) - 1)
            + ft_Sm * (np.exp(lambda_147_yr * date_guess) - 1)
            - total_He
        )
        f_date_prime = (
            ft_U238 * lambda_238_yr * np.exp(lambda_238_yr * date_guess)
            + ft_U235 * lambda_235_yr * np.exp(lambda_235_yr * date_guess)
            + ft_Th * lambda_232_yr * np.exp(lambda_232_yr * date_guess)
            + ft_Sm * lambda_147_yr * np.exp(lambda_147_yr * date_guess)
        )
        date_diff = date_guess
    
        while abs(date_diff) > tolerance:
            corrected_date = date_guess - f_date / f_date_prime
            date_diff = date_guess - corrected_date
            date_guess = corrected_date
            f_date = (
                ft_U238 * (np.exp(lambda_238_yr * date_guess) - 1)
                + ft_U235 * (np.exp(lambda_235_yr * date_guess) - 1)
                + ft_Th * (np.exp(lambda_232_yr * date_guess) - 1)
                + ft_Sm * (np.exp(lambda_147_yr * date_guess) - 1) 
                - total_He
            )
            f_date_prime = (
                ft_U238 * lambda_238_yr * np.exp(lambda_238_yr * date_guess)
                + ft_U235 * lambda_235_yr * np.exp(lambda_235_yr * date_guess)
                + ft_Th * lambda_232_yr * np.exp(lambda_232_yr * date_guess)
                + ft_Sm * lambda_147_yr * np.exp(lambda_147_yr * date_guess)
            )

        # convert to Ma
        corrected_date = corrected_date / 1000000
        
        return corrected_date

    def CN_profile(
            self, 
            diffs,  
            init_He, 
            eject=True, 
            produce=True,
            divide=False
            ):
        """
        Returns the 1D, spherical diffusion profiles for the bulk grain using the Crank-Nicolson diffusion function.

        Parameters
        ----------

        diffs: 1D array of floats
            Diffusivities for each time step. Diffusivities must have units of microns^2/s. Must be length of the relevant time-temperature (or step-heating) path.

        init_He: 1D array
            1D profile of alphas (in atoms/g) for lattice, must be length of nodes.
        
        eject: optional boolean
            Allows for a non-alpha ejected diffusion profile. Default is 'True', meaning the profile will be alpha ejected.
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True', meaning production will occur.
        
        divide: optional boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses. Default is 'False'.


        
        Returns
        ----------
        bulk_He_profile: list of floats
            The 1D radial profile of diffused He in the bulk grain

        total_bulk_He: float
            The total amount of bulk helium present in atoms per spherical volume (base of 1/(4/3 * Pi)) 

        """

        # convert 1D profiles to radial position profiles
        # reflects 1st node position as 0.5 * r_step from grain center

        if init_He is not None:
            init_He = init_He * self._r_vals

        if eject:
            bulk_He_profile = self.CN_diffusion(
                self._nodes, 
                self._r_step, 
                self._relevant_tT, 
                diffs,
                self._aej_U238, 
                self._aej_U235, 
                self._aej_Th, 
                self._aej_Sm, 
                init_He,
                produce,
                divide
            )
        else:
            bulk_He_profile = self.CN_diffusion(
                self._nodes, 
                self._r_step, 
                self._relevant_tT, 
                diffs,
                self._no_aej_U238, 
                self._no_aej_U235, 
                self._no_aej_Th, 
                self._no_aej_Sm, 
                init_He,
                produce,
                divide
            )

        bulk_He_profile, total_bulk_He = self._integrate_profile(bulk_He_profile)

        return bulk_He_profile, total_bulk_He
    
    def _integrate_profile(self, bulk_He_profile):
        """
        Private helper function that checks for zero helium concentration, then uses Romberg integration to calculate total helium in the bulk grain.

        Parameters
        ----------

        bulk_He_profile: 1D array of floats
            Concentration profile of diffused He in the bulk grain

        Returns
        -------

        bulk_He_profile: 1D array of floats
            Concentration profile, zeroed out if concentration went negative
        
        total_bulk_He: float
            Total helium in atoms per spherical volume (base of 1/(4/3 * Pi))
            
        """
        if np.min(bulk_He_profile) < -bulk_He_profile[0] * 0.5:
            return np.zeros_like(bulk_He_profile), 0.0

        integral = bulk_He_profile * 4 * np.pi * self._r_vals**2
        total_bulk_He = romb(integral, self._r_step) / ((4/3) * np.pi)

        return bulk_He_profile, total_bulk_He

class zircon(crystal):
    def __init__(
            self, radius, log2_nodes, relevant_tT, rho_r_array, U_ppm, Th_ppm, Sm_ppm=0
            ):
        """
        Constructor for zircon class, a sub class of the crystal super class.

        Parameters
        ----------

        radius: float
            Radius of a sphere with equivalent surfacea area to volume ratio as grain (in micrometers)

        log2_nodes: int
            Number of nodes for 1D finite difference diffusion solver is equivalent to 2^log2_nodes + 1 (necessary for Romberg integration)
        
        relevant_tT: 2D array of floats
            Discretized time-temperature path

        rho_r_array: 2D array of floats
            Matrix of dimensionless track density for the time-temperature path relevant_tT. Assumes that the # of rows in rho_r_array and relevant_tT are equivalent
        
        U_ppm: float
            Concentration of uranium in zircon (in ppm)
        
        Th_ppm: float
            Concentration of thorium in zircon (in ppm)
        
        Sm_ppm: optional float
            Concentration of samarium in zircon (in ppm), default is 0    
    
        """
        super().__init__()
        self._radius = radius
        self._log2_nodes = log2_nodes
        self._nodes = 2**log2_nodes + 1
        # grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center
        self._r_step = radius / (self._nodes + 0.5)
        self._r_vals = (np.arange(self._nodes) + 0.5) * self._r_step
        self._U_ppm = U_ppm
        self._Th_ppm = Th_ppm
        self._Sm_ppm = Sm_ppm
        self._relevant_tT = relevant_tT
        self._rho_r_array = rho_r_array

        # compute alpha ejection profiles
        self._aej_U238, self._aej_U235, self._aej_Th, self._aej_Sm,self._corr_factors = self.zircon_alpha_ejection()
        self._no_aej_U238, self._no_aej_U235, self._no_aej_Th, self._no_aej_Sm, self._no_corr_factors = self.zircon_no_ejection()

    def get_radius(self):
        return self._radius
    
    def get_log2_nodes(self):
        return self._log2_nodes
    
    def get_nodes(self):
        return self._nodes
    
    def get_r_step(self):
        return self._r_step
    
    def get_r_vals(self):
        return self._r_vals
    
    def get_U_ppm(self):
        return self._U_ppm
    
    def get_Th_ppm(self):
        return self._Th_ppm
    
    def get_Sm_ppm(self):
        return self._Sm_ppm
    
    def get_relevant_tT(self):
        return self._relevant_tT
    
    def get_rho_r_array(self):
        return self._rho_r_array
    
    def get_aej(self):
        return self._aej_U238, self._aej_U235, self._aej_Th, self._aej_Sm, self._corr_factors
    
    def get_no_aej(self):
        return self._no_aej_U238, self._no_aej_U235, self._no_aej_Th, self._no_aej_Sm, self._no_corr_factors
    
    def set_rho_r_array(self, new_rho_r_array):
        self._rho_r_array = new_rho_r_array

    def set_relevant_tT(self, new_tT):
        self._relevant_tT = new_tT
    
    def zircon_alpha_ejection(self):
        return self.alpha_ejection(
            self._radius,
            self._nodes,
            self._r_step,
            self._U_ppm,
            self._Th_ppm,
            self._Sm_ppm,
            'zircon',
        )
    
    def zircon_no_ejection(self):
        return self.alpha_ejection(
            self._radius,
            self._nodes,
            self._r_step,
            self._U_ppm,
            self._Th_ppm,
            self._Sm_ppm,
            'none',
        )
        
    def guenthner_damage(self, initial_damage=0.0):
        """
        Calculates the amount of radiation damage at each time step of class variable relevant_tT using the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01). The damage amounts can then be directly used to calculate diffusivities at each time step.

        Parameters
        ----------
        initial_damage : optional float
            Pre-existing radiation damage at the start of the tT path, in units of alpha/g. Default is 0.0 (no pre-existing damage).
        
        Returns
        -------
        
        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT. Length of damage is one less than the number of rows in relevant_tT (because last time step is 0).
        
        """
        U238_atom = self._U_ppm * U238_ppm_atom
        U235_atom = self._U_ppm * U235_ppm_atom
        Th_atom = self._Th_ppm * Th_ppm_atom
        relevant_tT = self._relevant_tT
        rho_r_array = self._rho_r_array

        # compute the exponential arrays for all time steps at once
        time = relevant_tT[:, 0]                        
        exp_238 = np.exp(lambda_238 * time)    
        exp_235 = np.exp(lambda_235 * time)    
        exp_232 = np.exp(lambda_232 * time)    

        # calculate dose in alpha/g at each time step
        alpha_i = (
            8 * U238_atom * (exp_238[:-1] - exp_238[1:])
            + 7 * U235_atom * (exp_235[:-1] - exp_235[1:])
            + 6 * Th_atom * (exp_232[:-1] - exp_232[1:]) 
        )

        # reverse it so it's in the correct order as tT (old -> young)
        alpha_i = alpha_i[::-1]

        # add any initial damage to the first index in alpha_i
        alpha_i[0] += initial_damage

        # multiple each row of the rho_r_array by alpha_i
        n = np.size(relevant_tT, 0)
        alpha_e_array = rho_r_array[:n - 1, :n - 1] * alpha_i
        
        # sum the columns of each row of alpha_e_array to get the total damage at each time step
        damage = np.sum(alpha_e_array, axis=1)

        return damage
    
    def guenthner_diffs(self, damage):
        """
        Calculates the diffusivity at each time step of class variable relevant_tT using the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01).

        Parameters
        ----------

        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT. Length must be one less than the number of rows in relevant_tT (because last time step is 0).

        
        Returns
        -------
        
        diff_array: 1D array of floats
            Diffusivities as a function of damage at each time step of relevant_tT
        
        """
        # radius to micrometers
        radius = self._radius
        # time in seconds, temp in K
        relevant_tT = self._relevant_tT

        # Guenthner et al. (2013) diffusion equation parameters, Eas are in kJ/mol, D0s converted to microns2/s
        Ea_l = 165.0
        D0_l = 193188.0 * 10**8 
        D0_N17 = 0.006367 * 10**8
        Ea_N17 = 70.74 

        # g amorphized per alpha event
        Ba = 5.48e-19
        interconnect = 3.0

        # empirical constraints for damage chains from Ketcham et al. (2013) (https://doi.org/10.2138/am.2013.4249) 
        # track surface to volume ratio in nm^-1 
        SV = 1.669 
        # mean unidirectional length of travel until damage zone in a zircon with 1e14 alphas/g, in nm
        lint_0 = 45920.0

        # mean temperature (in K) across each step of tT
        mean_temp = (relevant_tT[:-1, 1] + relevant_tT[1:, 1]) / 2

        # shared exponential terms computed once for the whole damage array
        exp_fa_tort = np.exp(-Ba * damage)
        exp_fa_prime = np.exp(-Ba * damage * interconnect)

        # safety check for fully amorphous regime where exp_fa_prime underflows to zero
        safe_exp_fa_prime = np.where(exp_fa_prime > 0, exp_fa_prime, np.finfo(float).tiny)

        # tau array
        # prevent divide by zero when damage is zero or very small
        # the 1.0 is dummy value, which will be masked out by final np.where
        tau = np.where( damage >= 1e14, (lint_0 / (4.2 / ((1 - exp_fa_tort) * SV) - 2.5))**2, 1.0)

        # Arrhenius equations for D_l and D_N17 arrays
        D_l = D0_l * np.exp(-Ea_l / (gas_constant * mean_temp))
        D_N17 = D0_N17 * np.exp(-Ea_N17 / (gas_constant * mean_temp))

        # calculate diffusivities at each time step using equation 8 in Guenthner et al. (2013), units are in micrometers2/s
        fc_prime_term = safe_exp_fa_prime / ((1/tau) * (D_l / (radius * safe_exp_fa_prime)**2))
        fa_prime_term = (1 - safe_exp_fa_prime) / (D_N17 / (radius * (1 - safe_exp_fa_prime))**2)
        diff_damaged = radius**2 / (fa_prime_term + fc_prime_term)

        # minimal diffusivity allowed equivalent to zircons with 1e14 alphas/g, prevents divide by zero in diffusivity calculation
        diff_undamaged = D_l

        #determine if we're below the 1e14 alphas/g cutoff
        diff_array = np.where(damage >= 1e14, diff_damaged, diff_undamaged)
    
        return diff_array
            
    def zirc_date(self, diff_model, dam_model, init_He = None, init_damage = 0.0, eject=True, produce=True, divide=False):
        """
        Zircon (U-Th)/He date calculator. First calculates the diffusivity at each time step of class variable relevant_tT using various parameterizations. Current available diffusion models include Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01), and Guenthner et al. (XXXX). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally, the parent class method He_date is called to convert the He profile to a (U-Th)/He date.

        Parameters
        ----------

        diff_model: string
            Diffusion model, current choices are 'guenthner' for the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01), and 'mp_diffusion' for the parameterization of Guenthner et al. (xxxx)

        dam_model: string
            Damage annealing model, current choices are 'guenthner' for the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01)

        init_He: optional 1D array of floats
            User defined initial profile of helium concentration (atoms) in 1D from grain center to rim. It should be noted that lower-level diffusion solvers require units of concentration per radial position (i.e. in u, where u = concentration * radial position), but the conversion is done internally with the expectation that users will more commonly have initial profiles in terms of simply concentration. Default is 'None'.

        init_damage: optional float
            Pre-existing radiation damage at the start of the tT path, in units of alpha/g. Default is 0.0 (no pre-existing damage)
        
        eject: optional boolean
            Allows for a non-alpha ejected diffusion profile. Default is 'True', meaning the profile will be alpha ejected.
            
        produce: optional boolean
            Allows for no alpha production during diffusion. Default is 'True', meaning production will occur.
        
        divide: optional boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses. Default is 'False'.
        
        Returns
        -------
        
        date: float
            Zircon (U-Th)/He date, corrected for alpha ejection

        final_damage: float
            The total amount of damage (in units of alphas/g) at the end of the time-temperature history.

        He_profile: 1D array of floats
            The 1D radial profile of diffused He (units of atoms).

        total_He: float
            The total amount of helium in atoms.
        
        """
        # calculate damage levels using guenthner_damage function
        if dam_model == 'guenthner':
            damage = self.guenthner_damage(init_damage)
        
        # get diff_array from guenthner_diffs method
        if diff_model == 'guenthner':
            diff_array = self.guenthner_diffs(damage)
        elif diff_model == 'mp_diffusion':
            diff_array = self.mp_diffs(damage)[2]
            
        # send it all to the CN_profile method
        He_profile, total_He = self.CN_profile(
            diff_array,
            init_He,
            eject,
            produce,
            divide
        )

        # calculate date
        
        final_damage = damage[-1]
        if total_He == 0:
            return 0.0, final_damage, He_profile, 0.0        

        date = self.He_date(total_He, self._corr_factors)
        return date, final_damage, He_profile, total_He
    
    def mp_diffs(self, damage):
        """
        Calculates the diffusivity at each time step of class variable relevant_tT using the multi-path parameterization of Guenthner et al. (XXXX)

        Parameters
        ----------

        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT. Length of damage must be one less than the number of rows in relevant_tT (because last time step is 0).

            
        Returns
        -------
        
        fast_diff_array: 1D array of floats
            Array of the fast path diffusivities as a function of damage at each time step of relevant_tT. Length is one less than the number of rows in relevant_tT (because last time step is 0). Diffusivities are in micrometers**2/s.

        lat_diff_array: 1D array of floats
            Array of the lattice diffusivities as a funcition of damage at each time step of relevant_tT. Length is one less than the number of rows in relevant_tT (because last time step is 0). Diffusivities are in micrometers**2/s.

        bulk_diff_array: 1D array of floats
            Array of the bulk diffusivities as a function of damage at each time step of relevant_tT. Length is one less than the number of rows in relevant_tT (because last time step is 0). Diffusivities are in micrometers**2/s.
        
        """
        # temp in K
        relevant_tT = self._relevant_tT

        # mass of amorphous material produced per alpha event (g/alpha)
        # Palenik et al. 2003
        B_a = 5.48e-19

        # surface area to volume ratio of damage capsule (cm-1)
        SV = 1.669

        # Guenthner et al. (2026) diffusion equation parameters

        D0_z = 867.0 * 10**8        # micron^2/s
        D0_sc = 0.00702 * 10**8     # micron^2/s
        Ea_z = 168.72               # kJ/mol
        Ea_sc = 70.74               # kJ/mol
        Ea_trap = 27.1              # kJ/mol
        l_int_0 = 4510.0            # nm
        gamma = 0.00146
        kappa_1 = 1e-3
        k_star = 1.385e-11
        n_sc = 72.76
        n_t = 0.7398

        # mean temperature (in K) across each step of tT
        mean_temp = (relevant_tT[:-1, 1] + relevant_tT[1:, 1]) / 2

        # tort parameter arrays
        f_a_DI = 1 - np.exp(-B_a * damage)
        l_int = (4.2 / (f_a_DI * SV)) - 2.5
        tau = (1 + (l_int_0 / l_int))**n_t
        D0_v = D0_z * (1 / tau)

        # trap parameter array
        psi = (gamma * f_a_DI * np.exp(Ea_trap/(gas_constant * mean_temp))) + 1

        # lattice (volume) diffusivity array
        lat_diff_array = D0_v * np.exp(-Ea_z / (gas_constant * mean_temp)) / psi

        # fast path diffusivity array
        fast_diff_array = D0_sc * np.exp(-Ea_sc / (gas_constant * mean_temp))

        # fraction amorphous
        f = 1 - ((1 + B_a * damage) * np.exp(-(B_a * damage)))**n_sc

        # protect against values of f = 1, which blows up kappa_2
        f = np.clip(f, 0, 1 - 1e-15)

        # solve for kappa_2
        kappa_2 = -kappa_1 * (k_star * f) / (1 - f)

        # bulk diffusivity array
        bulk_diff_array = (
            (1 / (kappa_1 - kappa_2)) * 
            (-kappa_2 * fast_diff_array + kappa_1 * lat_diff_array)
        )
        
        return fast_diff_array, lat_diff_array, bulk_diff_array

    def mp_profile(
            self, 
            diff_parameters, 
            tolerance, 
            init_fast_He, 
            init_lat_He, 
            eject=True, 
            produce=True,
            ):
        """
        Returns the 1D, spherical diffusion profiles for lattice, fast path, and the bulk grain using the multi-path diffusion function.

        Parameters
        ----------

        diff_parameters: dictionary of floats
            Fitted parameters for multi-path diffusion. Diffusivities must have units of microns^2/s

        tolerance: float
            Convergence criterion for iterative diffusion algorithm

        init_fast_He: 1D array
            1D profile of alphas (in atoms/g) for fast path, must be length of nodes. 

        init_lat_He: 1D array
            1D profile of alphas (in atoms/g) for lattice, must be length of nodes.
        
        eject: optional boolean
            Allows for a non-alpha ejected diffusion profile. Default is 'True', meaning the profile will be alpha ejected.
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True', meaning production will occur.

        
        Returns
        ----------
        bulk_He_profile: list of floats
            The 1D radial profile of diffused He in the bulk grain
        
        fast_He_profile: list of floats
            The 1D radial profile of diffused He in the fast pathways

        lat_He_profile: list of floats
            The 1D radial profile of diffused He in the lattice

        total_bulk_He: float
            The total amount of bulk helium present in atoms per spherical volume (base of 1/(4/3 * Pi)) 
 
        """

        # convert 1D profiles to radial position profiles
        # reflects 1st node position as 0.5 * r_step from grain center
        init_fast_He = init_fast_He * self._r_vals
        init_lat_He = init_lat_He  * self._r_vals

        if eject:
            bulk_He_profile, fast_He_profile, lat_He_profile = self.mp_diffusion(
                self._nodes, 
                self._r_step, 
                self._relevant_tT, 
                diff_parameters, 
                tolerance, 
                self._aej_U238, 
                self._aej_U235, 
                self._aej_Th, 
                self._aej_Sm, 
                init_fast_He,
                init_lat_He,
                produce,
            )
        else:
            bulk_He_profile, fast_He_profile, lat_He_profile = self.mp_diffusion(
                self._nodes, 
                self._r_step, 
                self._relevant_tT, 
                diff_parameters, 
                tolerance, 
                self._no_aej_U238, 
                self._no_aej_U235, 
                self._no_aej_Th, 
                self._no_aej_Sm, 
                init_fast_He,
                init_lat_He,
                produce,
            )

        bulk_He_profile, total_bulk_He = self._integrate_profile(bulk_He_profile)

        return bulk_He_profile, fast_He_profile, lat_He_profile, total_bulk_He


class apatite(crystal):
    def __init__(
            self, radius, log2_nodes, relevant_tT, rho_r_array, U_ppm, Th_ppm, Sm_ppm=0
            ):
        """
        Constructor for apatite class, a sub class of the crystal super class.

        Parameters
        ----------

        radius: float
            Radius of a sphere with equivalent surfacea area to volume ratio as grain (in micrometers)

        log2_nodes: int
            Number of nodes for 1D finite difference diffusion solver is equivalent to 2^log2_nodes + 1
        
        relevant_tT: 2D array of floats
            Discretized time-temperature path

        rho_r_array: 2D array of floats
            Matrix of dimensionless track density for the time-temperature path relevant_tT. Assumes that the # of rows in rho_r_array and relevant_tT are equivalent
        
        U_ppm: float
            Concentration of uranium in apatite (in ppm)
        
        Th_ppm: float
            Concentration of thorium in apatite (in ppm)
        
        Sm_ppm: optional float
            Concentration of samarium in apatite (in ppm), default is 0    
    
        """
        super().__init__()
        self._radius = radius
        self._log2_nodes = log2_nodes
        self._nodes = 2**log2_nodes + 1
        # grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center
        self._r_step = radius / (self._nodes + 0.5)
        self._r_vals = (np.arange(self._nodes) + 0.5) * self._r_step
        self._U_ppm = U_ppm
        self._Th_ppm = Th_ppm
        self._Sm_ppm = Sm_ppm
        self._relevant_tT = relevant_tT
        self._rho_r_array = rho_r_array

        # compute alpha ejection profiles
        self._aej_U238, self._aej_U235, self._aej_Th, self._aej_Sm, self._corr_factors = self.apatite_alpha_ejection()
        self._no_aej_U238, self._no_aej_U235, self._no_aej_Th, self._no_aej_Sm, self._no_corr_factors = self.apatite_no_ejection()

    def get_radius(self):
        return self._radius
    
    def get_log2_nodes(self):
        return self._log2_nodes
    
    def get_nodes(self):
        return self._nodes
    
    def get_r_step(self):
        return self._r_step
    
    def get_r_vals(self):
        return self._r_vals
    
    def get_U_ppm(self):
        return self._U_ppm
    
    def get_Th_ppm(self):
        return self._Th_ppm
    
    def get_Sm_ppm(self):
        return self._Sm_ppm
    
    def get_relevant_tT(self):
        return self._relevant_tT
    
    def get_rho_r_array(self):
        return self._rho_r_array
    
    def get_aej(self):
        return self._aej_U238, self._aej_U235, self._aej_Th, self._aej_Sm, self._corr_factors
    
    def get_no_aej(self):
        return self._no_aej_U238, self._no_aej_U235, self._no_aej_Th, self._no_aej_Sm, self._no_corr_factors
    
    def set_rho_r_array(self, new_rho_r_array):
        self._rho_r_array = new_rho_r_array

    def set_relevant_tT(self, new_tT):
        self._relevant_tT = new_tT
    
    def apatite_alpha_ejection(self):
        return self.alpha_ejection(
            self._radius,
            self._nodes,
            self._r_step,
            self._U_ppm,
            self._Th_ppm,
            self._Sm_ppm,
            'apatite',
        )
    
    def apatite_no_ejection(self):
        return self.alpha_ejection(
            self._radius,
            self._nodes,
            self._r_step,
            self._U_ppm,
            self._Th_ppm,
            self._Sm_ppm,
            'none',
        )
        
    def flowers_damage(self, initial_damage=0.0):
        """
        Calculates the amount of radiation damage at each time step of class variable relevant_tT using the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). The damage amounts can then be directly used to calculate diffusivities at each time step.

        Parameters
        ----------
        initial_damage : optional float
            Pre-existing radiation damage at the start of the tT path, in units of atoms/cc. Default is 0.0 (no pre-existing damage).

        Returns
        -------
        
        damage: 1D array of floats
            Total amount of damage at each time step of relevant_tT
        
        """
        # constants for the e_rho_s calculation, as defined in Flowers et al. (2009)
        eta_q = 0.91
        L = 0.000815
        
        # convert ppm to atoms/cc
        U235_vol = self._U_ppm * U235_ppm_atom * ap_density
        U238_vol = self._U_ppm * U238_ppm_atom * ap_density
        Th_vol = self._Th_ppm * Th_ppm_atom * ap_density
        Sm_vol = self._Sm_ppm * Sm_ppm_atom * ap_density
        
        relevant_tT = self._relevant_tT
        rho_r_array = self._rho_r_array

        # compute the exponential arrays for all time steps at once
        time = relevant_tT[:, 0]                        
        exp_238 = np.exp(lambda_238 * time)    
        exp_235 = np.exp(lambda_235 * time)    
        exp_232 = np.exp(lambda_232 * time)
        exp_147 = np.exp(lambda_147 * time) 

        # rho v array, atoms/cc, shape (n-1,)
        rho_v = (
            U238_vol * (exp_238[:-1] - exp_238[1:]) 
            + (7/8) * U235_vol * (exp_235[:-1] - exp_235[1:]) 
            + (6/8) * Th_vol * (exp_232[:-1] - exp_232[1:]) 
            + (1/8) * Sm_vol * (exp_147[:-1] - exp_147[1:]) 
        )

        # reverse it so it's in the correct order as tT (old -> young)
        rho_v = rho_v[::-1]

        # add any initial damage to the first index in rho_v
        rho_v[0] += initial_damage
        
        # sum the columns of each row of rho_r_array
        rho_r = np.sum(rho_r_array, axis=1)
        
        # e_rho_s calculation, shape (n-1,)
        n = np.size(relevant_tT, 0)
        damage = eta_q * L * (lambda_f / lambda_238) * rho_v * rho_r[:n-1]

        return damage
    
    def flowers_diffs(self, damage):
        """
        Calculates the diffusivity at each time step of class variable relevant_tT using the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

        Parameters
        ----------

        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT

        
        Returns
        -------
        
        diff_array: 1D array of floats
            Diffusivities as a function of damage at each time step of relevant_tT. Units are in micrometers2/s.
        
        """
        # Flowers et al. 2009 damage-diffusivity equation parameters
        omega = 1e-22
        psi = 1e-13
        E_trap = 34.0   # kJ/mol
        E_L = 122.3     # kJ/mol
        D0_L_a2 = np.exp(9.733)  # 1/s

        # convert D0_L_a2 to micrometers2/s using crystal radius
        D0_L = D0_L_a2 * self._radius**2

        relevant_tT = self._relevant_tT

        # mean temperature across each step in tT
        mean_temp = (relevant_tT[:-1, 1] + relevant_tT[1:, 1]) / 2

        #calculate diffusivities at each time step, equation 8 in Flowers et al. (2009), units are in micrometers2/s
        numerator = D0_L * np.exp(-E_L / (gas_constant * mean_temp))
        trap_factor = (psi * damage + omega * damage**3) * np.exp(E_trap / (gas_constant * mean_temp))
        diff_array = numerator / (trap_factor + 1)

        return diff_array
    
    def ap_date(self, diff_model, dam_model, init_He = None, init_damage = 0.0, eject=True, produce=True, divide=False):
        """
        Apatite (U-Th)/He date calculator. First calculates the diffusivity at each time step of class variable relevant_tT using various parameterizations. Current available diffusion models include Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally,the parent class method He_date is called to convert the He profile to a (U-Th)/He date.

        Parameters
        ----------

        diff_model: string
            Diffusion model, current choices are 'flowers' for the Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

        dam_model: string
            Damage annealing model, current choices are 'flowers' for the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

        init_He: optional 1D array of floats
            User defined initial profile of helium concentration (atoms) in 1D from grain center to rim. It should be noted that lower-level diffusion solvers require units of concentration per radial position (i.e. in u, where u = concentration * radial position), but the conversion is done internally with the expectation that users will more commonly have initial profiles in terms of simply concentration. Default is 'None'.
        
        init_damage: optional float
            Pre-existing radiation damage at the start of the tT path, in units of atoms/cc. Default is 0.0 (no pre-existing damage)
        
        eject: optional boolean
            Allows for a non-alpha ejected diffusion profile. Default is 'True', meaning the profile will be alpha ejected.
            
        produce: optional boolean
            Allows for no alpha production during diffusion. Default is 'True', meaning production will occur.
        
        divide: optional boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses. Default is 'False'.

           

        Returns
        -------
        
        date: float
            Apatite (U-Th)/He date, corrected for alpha ejection

        final_damage: float
            The total amount of damage (in units of spontaneous track density) at the end of the time-temperature history.

        He_profile: 1D array of floats
            The 1D radial profile of diffused He (units of atoms/g).
        
        total_He: float
            The total amount of helium in atoms.
        
        """
        # calculate damage levels using flowers_damage function
        if dam_model == 'flowers':
            damage = self.flowers_damage(init_damage)

        # calculate diffusivities using flowers_diffs function
        if diff_model == 'flowers':
            diff_list = self.flowers_diffs(damage)

        # send it all to the CN_profile method
        He_profile, total_He = self.CN_profile(
            diff_list,
            init_He,
            eject,
            produce,
            divide,
        )

        # calculate date
        
        final_damage = damage[-1]
        if total_He == 0:
            return 0.0, final_damage, He_profile, 0.0        

        date = self.He_date(total_He, self._corr_factors)
        return date, final_damage, He_profile, total_He
