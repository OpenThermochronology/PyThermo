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
from .core_solvers import _CN_diffusion_core, _mp_diffusion_core
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
        #unit conversion for U,Th,Sm inputs to atoms/g
        U238_atom = U_ppm * U238_ppm_atom
        U235_atom = U_ppm * U235_ppm_atom
        Th_atom = Th_ppm * Th_ppm_atom
        Sm_atom = Sm_ppm * Sm_ppm_atom

        #alpha stopping distances as reported in Ketcham et al. (2011) (https://doi.org/10.1016/j.gca.2011.10.011)
        #option for no alpha ejection   
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

        #alpha ejection profile arrays for each isotope
        #compute radial positions
        r_pos = (np.arange(nodes) + 0.5) * r_step

        #create masks for each isotope to ensure no divide by zero or very near zero
        mask_U238 = r_pos >= (radius - as_U238)
        mask_U235 = r_pos >= (radius - as_U235)
        mask_Th = r_pos >= (radius - as_Th)
        mask_Sm = r_pos >= (radius - as_Sm)

        #safe division, only divides where mask is true
        #still encounter RuntimeWarning because of the - r_pos... but these get discarded anyway with the mask, ignore these
        #U238
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

        #U235
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

        #Th
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

        #Sm
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
                
        #calculate alpha ejection correction factors for age equation
        outer = (np.arange(1, nodes + 1) * r_step)**3
        inner = (np.arange(nodes) * r_step)**3
        vol = outer - inner

        #weighted sums
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
                
        #calculate fts
        ft_U238 = nft_U238 / dft_U238
        ft_U235 = nft_U235 / dft_U235
        ft_Th = nft_Th / dft_Th
        ft_Sm = nft_Sm / dft_Sm

        #store in a dictionary
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
            1D profile of alphas for lattice, must be length of nodes. Default is None. Must be in terms of radial position. 
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True'.

        divide: optional boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses. Default is 'False'.

        Returns
        -------

        He_profile: array of floats
            The 1D radial profile of diffused He (units of atoms/g).

        """
        #damping parameters
        M = 100
        initial_damp = 20
        
        He_profile = _CN_diffusion_core(
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
            1D profile of alphas for fast path, must be length of nodes. Default is None. Must be in terms of radial position (see mp_profile function).
        
        init_lat_He: optional 1D array
            1D profile of alphas for lattice, must be length of nodes. Default is None. Must be in terms of radial position (see mp_profile function). 
        
        produce: optional boolean
            Allows for no alpha production during diffusion, useful for generating Arrhenius trends. Default is 'True'.


        Returns
        -------

        bulk_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the bulk grain
        
        fast_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the fast pathways

        lat_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the lattice

        """
        #unpack diff_parameters, D values have to be in units of micrometer**2/s
        D_sc = diff_parameters['D_sc']
        D_v = diff_parameters['D_v']
        kappa_1 = diff_parameters['kappa_1']
        kappa_2 = diff_parameters['kappa_2']
        f = diff_parameters['f']

        #damping parameters
        M = 100
        initial_damp = 20

        #set up initial arrays
        init_fast = np.asarray(init_fast_He, dtype=np.float64) if init_fast_He is not None else np.zeros(nodes)
        init_lat  = np.asarray(init_lat_He, dtype=np.float64) if init_lat_He is not None else np.zeros(nodes)

        bulk_He_profile, fast_He_profile, lat_He_profile = _mp_diffusion_core(
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

        #ensure damping steps don't produce negative times when dt_int is large, short-circuit if dt_int is large
        if dt_int > initial_damp + M:
            n_steps = int(dt / dt_int) + 1
            sub_tT = np.zeros((n_steps, 2))
            sub_tT[:, 0] = dt - np.arange(n_steps) * dt_int
            sub_tT[:, 1] = temp
            return sub_tT

        #set up iterative Newton-Raphson solver for eq 24 from Britz et al. (2003)
        fourier_calc = D * dt / r_step**2
        beta_guess = 1.2
        f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
        f_beta_prime = 2*fourier_calc*beta_guess - M*beta_guess**(M-1)
        tolerance = 1e-3

        beta_diff = beta_guess

        #iterate to solve
        while abs(beta_diff) > tolerance:
            beta = beta_guess - f_beta / f_beta_prime
            beta_diff = beta_guess - beta
            beta_guess = beta
            f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
            f_beta_prime = 2 * fourier_calc * beta_guess - M * beta_guess**(M - 1)

        #add on the initial damping steps
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

        #add on temperatures
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
        #get the guess to the 1000 year precision, date guess is in years
        tolerance = 1000.0
        date_guess = 100000000.0
        
        #get Fts from the corr_factors dictionary
        ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
        ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
        ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
        ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

        #set up Newton-Raphson equation
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

        #convert to Ma
        corrected_date = corrected_date / 1000000
        
        return corrected_date

    def romberg(self, integral, a, b, log2_nodes, r_step):
        """ 
        Implements Romberg's method for using the extended trapezoidal rule. Not called, here for instructional purposes. Adapated from Numerical Recipes, sections 4.2 and 4.3, Press et al. (2007) ISBN: 978-0-521-88068-8. Works on a vector of 2**n + 1 equally spaced samples of a function.

        Parameters
        ----------

        integral: 1D array or list of floats
            The "function" to integrate over, where the array index corresponds to each x value in f(x)

        a: float
            Starting position of integration, hardwired here to correspond to the postion of the first index in integral
        
        b: float
            End position of integration, hardwired here to correspond to the position of the last index in integral
                
        log2_nodes: int
            Value of log base 2 of nodes
        
        r_step: float
            Grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center

        Returns
        -------

        total_He: float
            The total amount of integrated helium in the crystal sphere

        """
        
        #highest node *index*, NOT total number of nodes (2**log2_nodes + 1) as used in other functions 
        nodes = 2**log2_nodes

        #store successive trapezoidal approximations in s and their relative step sizes in h
        h = np.zeros(log2_nodes + 1)
        s = np.zeros(log2_nodes + 1)

        h[0] = 1.0
        s[0] = 0.5 * (b - a) * (integral[0] + integral[-1])
        
        it = 1
        spacing = nodes // 2
        half_way = nodes // 2
        
        #main trapezoidal rule, successive "calls" are made here by for loop construction
        for i in range(1, log2_nodes + 1):
            sum = 0.0
            for j in range(half_way, nodes, spacing):
                sum = sum + integral[j]
            
            s[i] = 0.5 * (s[i - 1] + (b - a) * sum / it)
            it = it * 2

            #get area under curve at successive divisions (1/2, then 1/4, 3/4, then 1/8, 3/8, 5/8, 7/8, etc.)
            half_way = half_way // 2
            if i > 1:
                spacing = spacing // 2

            h[i] = 0.25 * h[i - 1]
        
        #extrapolate the refinements to zero stepsize using polynomial interpolation
        total_He = self.poly_interp(h, s, 5)

        #add in helium at the center of the sphere, which is a half-node spaced before the 0th index, NR equation 4.1.10
        total_He = total_He + 0.5 * r_step * (
            integral[0] * 55.0 / 24.0 
            - integral[1] * 59.0 / 24.0 
            + integral[2] * 37.0 / 24.0 
            - integral[3] * 9.0 / 24.0
        )
    
        return total_He

    def poly_interp(self, xa, ya, jl, x=0):
        """ 
        Polynomial interpolation method (Neville's algorithm), adapted from Numerical Recipes section 3.2, Press et al. (2007) ISBN: 978-0-521-88068-8. Given a starting value, x, two vectors of x values and f(x) solutions, use polynomials up to order jl-1 to interpolate and accumulate the rest of the y values. Not called, here for instructional purposes.

        Parameters
        ----------

        xa: 1D array of floats
            Vector of x values
        
        ya: 1D array of floats
            Vector of y = f(x) solutions

        jl: int
            Polynomial order plus one for interpolation

        x: optional int
            Starting value for the interpolation, default is 0

        Returns
        -------

        y: float
            Value of y, accumulated over course through the tableau
        
        """
        #set up the tableau of c's and d's 
        c = np.zeros(jl)
        d = np.zeros(jl)

        dif = abs(x - xa[0])
        #find the index ns of the closest table entry
        for i in range(jl):
            dift = abs(x - xa[i])
            if dift < dif:
                ns = i
                dif = dift
            
            c[i] = ya[i]
            d[i] = ya[i]
        ns = ns - 1

        #initial approximation of y
        y = ya[ns]

        #now loop  over the c's and d's for each column of the tableau
        for m in range(1, jl):
            for i in range(jl - m):
                ho = xa[i] - x
                hp = xa[i + m] - x
                w = c[i + 1] - d[i]
                den = ho - hp

                #this error can occur only if two input xa's are indentical (to within roundoff)
                try:
                    den = w / den
                except ZeroDivisionError:
                    print('Error in poly_interp: two values of xa were identical')

                #update the c's and d's
                d[i] = hp * den
                c[i] = ho * den
            
            #decide on which path (c or d) to take through the tableau, take the most "straight-line" route
            if 2 * ns < jl - m:
                dy = c[ns + 1]
            else:
                ns = ns - 1
                dy = d[ns]
            
            #last dy added could be returned as the error 
            y = y + dy
                
        return y

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
        self.__radius = radius
        self.__log2_nodes = log2_nodes
        self.__nodes = 2**log2_nodes + 1
        #grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center
        self.__r_step = radius / (self.__nodes + 0.5)
        self.__U_ppm = U_ppm
        self.__Th_ppm = Th_ppm
        self.__Sm_ppm = Sm_ppm
        self.__relevant_tT = relevant_tT
        self.__rho_r_array = rho_r_array

    def get_radius(self):
        return self.__radius
    
    def get_log2_nodes(self):
        return self.__log2_nodes
    
    def get_nodes(self):
        return self.__nodes
    
    def get_r_step(self):
        return self.__r_step
    
    def get_U_ppm(self):
        return self.__U_ppm
    
    def get_Th_ppm(self):
        return self.__Th_ppm
    
    def get_Sm_ppm(self):
        return self.__Sm_ppm
    
    def get_relevant_tT(self):
        return self.__relevant_tT
    
    def get_rho_r_array(self):
        return self.__rho_r_array
    
    def zircon_alpha_ejection(self):
        return self.alpha_ejection(
            self.__radius,
            self.__nodes,
            self.__r_step,
            self.__U_ppm,
            self.__Th_ppm,
            self.__Sm_ppm,
            'zircon',
        )
    
    def zircon_no_ejection(self):
        return self.alpha_ejection(
            self.__radius,
            self.__nodes,
            self.__r_step,
            self.__U_ppm,
            self.__Th_ppm,
            self.__Sm_ppm,
            'none',
        )
        
    def guenthner_damage(self):
        """
        Calculates the amount of radiation damage at each time step of class variable relevant_tT using the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01). The damage amounts can then be directly used to calculate diffusivities at each time step.
        
        Returns
        -------
        
        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT. Length of damage is one less than the number of rows in relevant_tT (because last time step is 0).
        
        """
        U238_atom = self.__U_ppm * U238_ppm_atom
        U235_atom = self.__U_ppm * U235_ppm_atom
        Th_atom = self.__Th_ppm * Th_ppm_atom
        relevant_tT = self.__relevant_tT
        rho_r_array = self.__rho_r_array

        #calculate dose in alpha/g at each time step
        alpha_i = [
            8
            * U238_atom
            * (
                np.exp(lambda_238 * relevant_tT[i, 0])
                - np.exp(lambda_238 * relevant_tT[i + 1,0])
            )
            + 7
            * U235_atom
            * (
                np.exp(lambda_235 * relevant_tT[i, 0])
                - np.exp(lambda_235 * relevant_tT[i + 1, 0])
            ) 
            + 6
            * Th_atom
            * (
                np.exp(lambda_232 * relevant_tT[i, 0])
                - np.exp(lambda_232 * relevant_tT[i + 1, 0])
            ) 
            for i in range(np.size(relevant_tT, 0) - 2, -1, -1)
        ]

        #multiple each row of the rho_r_array by alpha_i
        alpha_e_array = (
            rho_r_array[: np.size(relevant_tT, 0) - 1, : np.size(relevant_tT, 0) - 1] * alpha_i
        )

        #sum the columns of alpha_e_array to get the total damage at each time step
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
        
        diff_list: list of floats
            List of the diffusivities as a function of damage at each time step of relevant_tT
        
        """
        #radius to micrometers
        radius = self.__radius
        #time in seconds
        relevant_tT = self.__relevant_tT

        #Guenthner et al. (2013) diffusion equation parameters, Eas are in kJ/mol, D0s converted to microns2/s
        Ea_l = 165.0
        D0_l = 193188.0 * 10**8 
        D0_N17 = 0.0034 * 10**8
        Ea_N17 = 71.0 

        #g amorphized per alpha event
        Ba = 5.48e-19
        interconnect = 3.0

        #empirical constraints for damage chains from Ketcham et al. (2013) (https://doi.org/10.2138/am.2013.4249) 
        #track surface to volume ratio in nm^-1 
        SV = 1.669 
        #mean unidirectional length of travel until damage zone in a zircon with 1e14 alphas/g, in nm
        lint_0 = 45920.0

        #calculate diffusivities at each time step, modified equation 8 in Guenthner et al. (2013, units are in micrometers2/s; minimal diffusivity allowed equivalent to zircons with 1e14 alphas/g), prevents divide by zero in diffusivity calculation
        diff_list = [
            (
                radius**2 
                * (
                    (
                        (
                            radius**2
                            * np.exp(-Ba * damage[i] * interconnect) ** 3
                            * (
                                lint_0
                                / (4.2 / ((1 - np.exp(-Ba * damage[i])) * SV) - 2.5)
                            )
                                ** 2
                        ) 
                        / (
                            D0_l 
                            * np.exp(
                                -Ea_l
                                / (
                                    gas_constant
                                    * ((relevant_tT[i, 1] + relevant_tT[i + 1, 1]) / 2)
                                )
                            )
                        )
                    ) 
                    + (
                        (radius**2 * (1 - np.exp(-Ba * damage[i] * interconnect))) ** 3 
                        / (
                            D0_N17
                            * np.exp(
                                -Ea_N17
                                / (
                                    gas_constant
                                    * ((relevant_tT[i, 1] + relevant_tT[i + 1, 1]) / 2)
                                )
                            )
                        )
                    )
                )
                ** -1 
                if damage[i] >= 10**14 
                else radius**2 
                * (
                    (
                        radius**2
                        / (
                            D0_l 
                            * np.exp(
                                -Ea_l
                                / (
                                    gas_constant
                                    *((relevant_tT[i, 1] + relevant_tT[i + 1, 1]) / 2)
                                )
                            )
                        )
                    )
                )
                **-1
            )
            for i in range(len(damage))
        ]
    
        return diff_list
            
    def zirc_date(self, diff_model, dam_model):
        """
        Zircon (U-Th)/He date calculator. First calculates the diffusivity at each time step of class variable relevant_tT using various parameterizations. Current available diffusion models include Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01), and Guenthner et al. (XXXX). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally, the parent class method He_date is called to convert the He profile to a (U-Th)/He date.

        Parameters
        ----------

        diff_model: string
            Diffusion model, current choices are 'guenthner' for the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01), and 'mp_diffusion' for the parameterization of Guenthner et al. (xxxx)

        dam_model: string
            Damage annealing model, current choices are 'guenthner' for the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01)
        
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
        #calculate damage levels using guenthner_damage function
        if dam_model == 'guenthner':
            damage = self.guenthner_damage()
        
        #get diff_list from guenthner_diffs method
        if diff_model == 'guenthner':
            diff_list = self.guenthner_diffs(damage)
        elif diff_model == 'mp_diffusion':
            diff_list = self.mp_diffs(damage)[2]

        #create production lists that consider alpha ejection
        aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.zircon_alpha_ejection()

        #send it all to the CN_diffusion method
        He_profile = self.CN_diffusion(
            self.__nodes,
            self.__r_step,
            self.__relevant_tT,
            diff_list,
            aej_U238,
            aej_U235,
            aej_Th,
            aej_Sm,
        )

        #calculate date
        
        #check for zero helium concentration
        minimum_He = He_profile[0] * 0.5 * self.__r_step
        for i in range(self.__nodes):
            if He_profile[i] < minimum_He:
                minimum_He = He_profile[i]
        
        if minimum_He < -He_profile[0] * 0.5:
            total_He = 0
            final_damage = damage[-1]
            return total_He, final_damage, He_profile, total_He

        #convert He profile into a spherical function for integration
        integral = [
            He_profile[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
            for i in range(self.__nodes)
        ]

        #use Romberg integration to calculate total amount of He

        #homebrewed version (used only for instructional purposes)
        #start = 0.5 * self.__r_step
        #end = (self.__nodes - 0.5) * self.__r_step
        #total_He = self.romberg(integral, start, end, self.__log2_nodes, self.__r_step)

        #scipy version
        total_He = romb(integral, self.__r_step)

        #units in atoms per volume (base of 1/(4/3 * Pi))
        total_He = total_He / ((4 / 3) * np.pi)

        date = self.He_date(total_He, corr_factors)
        final_damage = damage[-1]
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
        fast_diff_array = np.zeros(len(damage))
        lat_diff_array = np.zeros(len(damage))
        bulk_diff_array = np.zeros(len(damage))
        
        #radius in micrometers
        radius = self.__radius
        
        #time in seconds, temp in C
        relevant_tT = self.__relevant_tT

        #mass of amorphous material produced per alpha event (g/alpha)
        #Palenik et al. 2003
        B_a = 5.48e-19

        #mean intercept length of zircon with dose of 1e14 alphas/g (nm)
        l_int_0 = 45920

        #surface area to volume ratio of damage capsule (cm-1)
        SV = 1.669

        #Guenthner et al. (XXXX) diffusion equation parameters, Eas are in kJ/mol, D0s converted to microns2/s
        D0_z = 1130 * 10**8 
        D0_sc = 0.0186 * 10**8
        Ea_z = 165.68
        Ea_sc = 70.74
        Ea_trap = 17.1
        gamma = 0.873
        kappa_1 = 0.101
        k_star = 9.33e-10
        n_sc = 7.038
        n_t = 0.37

        for i in range(len(damage)):
            #average temp between time steps
            temp_K = (relevant_tT[i, 1] + relevant_tT[i + 1, 1]) / 2
            #tort parameters
            f_a_DI = 1 - np.exp(-B_a * damage[i])**n_t
            l_int = (4.2 / (f_a_DI * SV)) - 2.5
            tau = (l_int_0 / l_int)**2
            D0_v = D0_z * (1 / tau)

            #trap parameters
            psi = (gamma * f_a_DI * np.exp(Ea_trap/(gas_constant * temp_K))) + 1

            #lattice (volume) diffusivity
            D_v = D0_v * np.exp(-Ea_z / (gas_constant * temp_K)) / psi
            lat_diff_array[i] = D_v

            #fast path (short-circuit) diffusivity
            D_sc = D0_sc * np.exp(-Ea_sc / (gas_constant * temp_K))
            fast_diff_array[i] = D_sc
            
            #fraction amorphous
            f = 1 - ((1 + B_a * damage[i]) * np.exp(-(B_a * damage[i])))**n_sc

            #treat kappa_1 as a constant, solve for kappa_2
            kappa_2 = -kappa_1 * (k_star * f) / (1 - f)

            #bulk diffusivity
            D_b = (
                (1 / (kappa_1 - kappa_2)) * 
                (-kappa_2 * D_sc + kappa_1 * D_v)
            )
            bulk_diff_array[i] = D_b
        
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

        #create production lists that consider (or don't) alpha ejection
        if eject:
            aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.zircon_alpha_ejection()
        else:
            aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.zircon_no_ejection()

        #convert 1D profiles to radial position profiles
        #reflects 1st node position as 0.5 * r_step from grain center
        init_fast_He = np.array([
            init_fast_He[i] * (i + 0.5) * self.__r_step for i in range(0, self.__nodes)
            ])
        
        init_lat_He = np.array([
            init_lat_He[i] * (i + 0.5) * self.__r_step for i in range(0, self.__nodes)
            ])

        bulk_He_profile, fast_He_profile, lat_He_profile = self.mp_diffusion(
            self.__nodes, 
            self.__r_step, 
            self.__relevant_tT, 
            diff_parameters, 
            tolerance, 
            aej_U238, 
            aej_U235, 
            aej_Th, 
            aej_Sm, 
            init_fast_He,
            init_lat_He,
            produce,
        )

        #use Romberg integration to calculate total amount of He for each profile
        #check for zero helium concentration
        minimum_bulk_He = np.min(bulk_He_profile)
        bulk_not_0 = True

        if minimum_bulk_He < -bulk_He_profile[0] * 0.5:
            total_bulk_He = 0
            bulk_not_0 = False
        

        #convert He profile into a spherical function for integration
        integral_bulk = [
            bulk_He_profile[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
            for i in range(self.__nodes)
        ]

        if bulk_not_0:
            total_bulk_He = romb(integral_bulk, self.__r_step)


        #units in atoms per volume (base of 1/(4/3 * Pi))
        total_bulk_He = total_bulk_He / ((4 / 3) * np.pi)

        return bulk_He_profile, fast_He_profile, lat_He_profile, total_bulk_He

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

            #create production lists that consider (or don't) alpha ejection
            if eject:
                aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.zircon_alpha_ejection()
            else:
                aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.zircon_no_ejection()

            #convert 1D profiles to radial position profiles
            #reflects 1st node position as 0.5 * r_step from grain center
            init_He = np.array([
                init_He[i] * (i + 0.5) * self.__r_step for i in range(0, self.__nodes)
                ])

            bulk_He_profile = self.CN_diffusion(
                self.__nodes, 
                self.__r_step, 
                self.__relevant_tT, 
                diffs,
                aej_U238, 
                aej_U235, 
                aej_Th, 
                aej_Sm, 
                init_He,
                produce,
                divide
            )

            #use Romberg integration to calculate total amount of He for each profile
            #check for zero helium concentration
            minimum_bulk_He = np.min(bulk_He_profile)
            bulk_not_0 = True

            if minimum_bulk_He < -bulk_He_profile[0] * 0.5:
                total_bulk_He = 0
                bulk_He_profile[:] = 0
                bulk_not_0 = False
            

            #convert He profile into a spherical function for integration
            integral_bulk = [
                bulk_He_profile[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
                for i in range(self.__nodes)
            ]

            if bulk_not_0:
                total_bulk_He = romb(integral_bulk, self.__r_step)


            #units in atoms per volume (base of 1/(4/3 * Pi))
            total_bulk_He = total_bulk_He / ((4 / 3) * np.pi)

            return bulk_He_profile, total_bulk_He

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
        self.__radius = radius
        self.__log2_nodes = log2_nodes
        self.__nodes = 2**log2_nodes + 1
        #grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center
        self.__r_step = radius / (self.__nodes + 0.5) 
        self.__U_ppm = U_ppm
        self.__Th_ppm = Th_ppm
        self.__Sm_ppm = Sm_ppm
        self.__relevant_tT = relevant_tT
        self.__rho_r_array = rho_r_array

    def get_radius(self):
        return self.__radius
    
    def get_log2_nodes(self):
        return self.__log2_nodes
    
    def get_nodes(self):
        return self.__nodes
    
    def get_r_step(self):
        return self.__r_step
    
    def get_U_ppm(self):
        return self.__U_ppm
    
    def get_Th_ppm(self):
        return self.__Th_ppm
    
    def get_Sm_ppm(self):
        return self.__Sm_ppm
    
    def get_relevant_tT(self):
        return self.__relevant_tT
    
    def get_rho_r_array(self):
        return self.__rho_r_array
    
    def apatite_alpha_ejection(self):
        return self.alpha_ejection(
            self.__radius,
            self.__nodes,
            self.__r_step,
            self.__U_ppm,
            self.__Th_ppm,
            self.__Sm_ppm,
            'apatite',
        )
    
    def apatite_no_ejection(self):
        return self.alpha_ejection(
            self.__radius,
            self.__nodes,
            self.__r_step,
            self.__U_ppm,
            self.__Th_ppm,
            self.__Sm_ppm,
            'none',
        )
        
    def flowers_damage(self):
        """
        Calculates the amount of radiation damage at each time step of class variable relevant_tT using the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). The damage amounts can then be directly used to calculate diffusivities at each time step.

        Returns
        -------
        
        damage: list of floats
            List of total amount of damage at each time step of relevant_tT
        
        """
        # constants for the e_rho_s calculation, as defined in Flowers et al. (2009)
        eta_q = 0.91
        L = 0.000815
        
        #convert ppm to atoms/cc
        U235_vol = self.__U_ppm * U235_ppm_atom * ap_density
        U238_vol = self.__U_ppm * U238_ppm_atom * ap_density
        Th_vol = self.__Th_ppm * Th_ppm_atom * ap_density
        Sm_vol = self.__Sm_ppm * Sm_ppm_atom * ap_density
        
        relevant_tT = self.__relevant_tT
        rho_r_array = self.__rho_r_array

        #rho v calculation, atoms/cc
        rho_v = [
            U238_vol
            * (
                np.exp(lambda_238 * relevant_tT[i, 0])
                - np.exp(lambda_238 * relevant_tT[i + 1, 0])
            ) 
            + (7 / 8)
            * U235_vol
            * (
                np.exp(lambda_235 * relevant_tT[i, 0])
                - np.exp(lambda_235 * relevant_tT[i + 1, 0])
            ) 
            + (6 / 8)
            * Th_vol
            * (
                np.exp(lambda_232 * relevant_tT[i, 0])
                - np.exp(lambda_232 * relevant_tT[i + 1, 0])
            ) 
            + (1 / 8)
            * Sm_vol
            * (
                np.exp(lambda_147 * relevant_tT[i, 0])
                -np.exp(lambda_147 * relevant_tT[i + 1, 0])
            ) 
            for i in range(np.size(relevant_tT, 0) - 2, -1, -1)
        ]
        
        #sum the columns of each row of rho_r_array
        rho_r = np.sum(rho_r_array, axis=1)
        
        #e_rho_s calculation
        damage = [
            eta_q * L * (lambda_f / lambda_238) * rho_v[i] * rho_r[i] 
            for i in range(np.size(relevant_tT, 0) - 1)
        ]
        
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
        
        diff_list: list of floats
            List of the diffusivities as a function of damage at each time step of relevant_tT. Units are in micrometers2/s.
        
        """
        #Flowers et al. 2009 damage-diffusivity equation parameters
        omega = 10**-22
        psi = 10**-13
        E_trap = 34.0  # kJ/mol
        E_L = 122.3  # kJ/mol
        D0_L_a2 = np.exp(9.733)  # 1/s

        #convert D0_L_a2 to micrometers2/s using crystal radius
        D0_L = D0_L_a2 * self.__radius**2

        relevant_tT = self.__relevant_tT

        #calculate diffusivities at each time step, equation 8 in Flowers et al. (2009), units are in micrometers2/s
        diff_list = [
            (
                D0_L 
                * np.exp(
                    -E_L
                    / (gas_constant * 0.5 * (relevant_tT[i, 1] + relevant_tT[i + 1,1]))
                    )
            )
            / (
                (
                    (psi * damage[i] + omega * damage[i] ** 3) 
                    * np.exp(
                        E_trap
                        / (
                            gas_constant
                            * 0.5
                            * (relevant_tT[i, 1] + relevant_tT[i + 1, 1])
                        )
                    )
                ) 
                + 1
            ) 
            for i in range(len(damage))
        ]

        return diff_list
    
    def ap_date(self, diff_model, dam_model):
        """
        Apatite (U-Th)/He date calculator. First calculates the diffusivity at each time step of class variable relevant_tT using various parameterizations. Current available diffusion models include Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally,the parent class method He_date is called to convert the He profile to a (U-Th)/He date.

        Parameters
        ----------

        diff_model: string
            Diffusion model, current choices are 'flowers' for the Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

        dam_model: string
            Damage annealing model, current choices are 'flowers' for the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

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
        #calculate damage levels using flowers_damage function
        if dam_model == 'flowers':
            damage = self.flowers_damage()

        #calculate diffusivities using flowers_diffs function
        if diff_model == 'flowers':
            diff_list = self.flowers_diffs(damage)

        #create production lists that consider alpha ejection
        aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.apatite_alpha_ejection()

        #send it all to the CN_diffusion method
        He_profile = self.CN_diffusion(
            self.__nodes,
            self.__r_step,
            self.__relevant_tT,
            diff_list,
            aej_U238,
            aej_U235,
            aej_Th,
            aej_Sm,
            )

        #calculate date

        #check for zero helium concentration
        minimum_He = He_profile[0] * 0.5 * self.__r_step
        for i in range(self.__nodes):
            if He_profile[i] < minimum_He:
                minimum_He = He_profile[i]
        
        if minimum_He < -He_profile[0] * 0.5:
            total_He = 0
            final_damage = damage[-1]
            return total_He, final_damage, He_profile, total_He
        
        #convert He profile into a spherical function for integration
        integral = [
            He_profile[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
            for i in range(self.__nodes)
        ]

        #use Romberg integration algorithm to calculate total amount of He

        #homebrewed version (used only for instructional purposes)
        #start = 0.5 * self.__r_step
        #end = (self.__nodes - 0.5) * self.__r_step
        #total_He = self.romberg(integral, start, end, self.__log2_nodes, self.__r_step)

        #scipy version
        total_He = romb(integral, self.__r_step)
        
        #units in atoms per volume (base of 1/(4/3 * Pi))
        total_He = total_He / ((4 / 3) *np.pi)

        date = self.He_date(total_He, corr_factors)
        final_damage = damage[-1]

        return date, final_damage, He_profile, total_He

    def CN_profile(
                self, 
                diffs,  
                init_He, 
                eject=True, 
                produce=True,
                divide=False,
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

            #create production lists that consider (or don't) alpha ejection
            if eject:
                aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.apatite_alpha_ejection()
            else:
                aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = self.apatite_no_ejection()

            #convert 1D profiles to radial position profiles
            #reflects 1st node position as 0.5 * r_step from grain center
            init_He = np.array([
                init_He[i] * (i + 0.5) * self.__r_step for i in range(0, self.__nodes)
                ])

            bulk_He_profile = self.CN_diffusion(
                self.__nodes, 
                self.__r_step, 
                self.__relevant_tT, 
                diffs,
                aej_U238, 
                aej_U235, 
                aej_Th, 
                aej_Sm, 
                init_He,
                produce,
                divide
            )

            #use Romberg integration to calculate total amount of He for each profile
            #check for zero helium concentration
            minimum_bulk_He = np.min(bulk_He_profile)
            bulk_not_0 = True

            if minimum_bulk_He < -bulk_He_profile[0] * 0.5:
                total_bulk_He = 0
                bulk_He_profile[:] = 0
                bulk_not_0 = False
            

            #convert He profile into a spherical function for integration
            integral_bulk = [
                bulk_He_profile[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
                for i in range(self.__nodes)
            ]

            if bulk_not_0:
                total_bulk_He = romb(integral_bulk, self.__r_step)


            #units in atoms per volume (base of 1/(4/3 * Pi))
            total_bulk_He = total_bulk_He / ((4 / 3) * np.pi)

            return bulk_He_profile, total_bulk_He
    