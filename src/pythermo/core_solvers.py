"""
core_solvers.py

Core methods for solving the diffusion and annealing equations. Uses Numba's jit compiler to achieve near-compiled language speeds on the main bottelnecks in the rest of the pythermo suite.

"""
from .constants import (
    np, 
    lambda_238,  
    lambda_235, 
    lambda_232, 
    lambda_147,
)
from numba import jit

@jit(nopython=True)
def _divide_tT(D, dt, temp, r_step, M, initial_damp):
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

@jit(nopython=True)
def _tridiag(a, diagonal, c, d, nodes):
    """
    Helper function for implementing the Thomas algorithm to solve tridiagonal systems to make CN and mp diffusion routines Numba compatible. Adapted for python from Numerical Recipes, section 2.4, Press et al. (2007) ISBN: 978-0-521-88068-8
    
    Parameters
    ----------

    a: 1D array of floats
        The sub-diagonal of the tridiagonal matrix. All ones except a[-1], which is 0 (zero boundary condition)
    
    diagonal: 1D array of floats
        The diagonal fo the tridiagonal matrix
    
    c: 1D array of floats
        The supra-diagonal of the tridiagonal matrix. All ones except c[0], which is 0 (zero flux boundary condition)
    
    d: 1D array of floats
        The right-hand side vector of the tridiagonal system: Ax = d

    nodes: int
        Total number of nodes in the diffusion solver (tridiagonal matrix is nodes x nodes)

    Returns
    -------

    x: 1D array of floats
        The solution vector for the tridagonal system: Ax = d
    
    """
    c_prime = np.zeros(nodes)
    d_prime = np.zeros(nodes)
    x = np.zeros(nodes)

    #forward sweep
    c_prime[0] = c[0] / diagonal[0]
    d_prime[0] = d[0] / diagonal[0]

    for i in range(1, nodes):
        denom = diagonal[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    #back substitution
    x[-1] = d_prime[-1]
    for i in range(nodes - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x

@jit(nopython=True)
def _CN_diffusion_core( 
            nodes, 
            r_step, 
            tT_path, 
            diffs, 
            aej_U238, 
            aej_U235, 
            aej_Th, 
            aej_Sm, 
            init_He, 
            allow,
            divide,
            M,
            initial_damp
            ):
        """ 
        Core function for solving the diffusion equation with production along a 1D radial profile using the Crank-Nicoloson finite difference scheme. Uses jit decorator. Assumes no flux across the center, or inner boundary, node (Neumann boundary condition) and zero concentration along the outer boundary (Dirichlet boundary condition). Original solution and set-up for the algorithm comes from Ketcham (2005) (https://doi.org/10.2138/rmg.2005.58.11), specifically equation 21, and equations 22 and 26 for boundary conditions.
        
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
        
        init_He: 1D array
            1D profile of alphas for lattice, must be length of nodes. Default is an array of zeros. Must be in terms of radial position. 
        
        allow: float
            Allows for alpha production (equals 1.0) or no alpha production (equals 0.0) during diffusion, useful for generating Arrhenius trends.

        divide: boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses.

        M: int
            Number of sub-divisions for the exponentially increasing damping function.

        initial_damp: int
            Number of sub-divions for the initial damping.


        Returns
        -------

        He_profile: array of floats
            The 1D radial profile of diffused He (units of atoms/g).

        """
        #set up arrays for tridiagonal matrix
        #u is the coordinate transform vector
        #u = vr, v is the He profile, r is radius

        u = init_He.copy()

        #setup diagonal and d (RHS vector, Ax = d)
        diagonal = np.zeros(nodes)
        d = np.zeros(nodes)

        #a is sub-, and c is supra-diagonal
        a = np.ones(nodes)
        a[-1] = 0.0
        c = np.ones(nodes)
        c[0] = 0.0

        r_positions = (np.arange(nodes) + 0.5) * r_step

        #step through time from old to young
        for i in range(tT_path.shape[0] - 1):

            dt_int = tT_path[i, 0] - tT_path[i + 1, 0]
            fourier = (diffs[i] * dt_int) / r_step**2

            #subdivide the time step if necessary
            if fourier > 0.5 and divide:
                temp = (tT_path[i, 1] + tT_path[i + 1, 1]) / 2
                sub_tT_path = _divide_tT(diffs[i], dt_int, temp, r_step, M, initial_damp)
            else:
                sub_tT_path = tT_path[i:i + 2, :]
           
            for j in range(sub_tT_path.shape[0] - 1):
                t_old = sub_tT_path[j, 0]
                t_young = sub_tT_path[j + 1, 0]
                dt = t_old - t_young
                beta = (2.0 * r_step**2) / (diffs[i] * dt)

                all_alphas = (
                8.0
                * aej_U238
                * (np.exp(lambda_238 * t_old) - np.exp(lambda_238 * t_young)) 
                + 7.0
                * aej_U235
                * (np.exp(lambda_235 * t_old) - np.exp(lambda_235 * t_young)) 
                + 6.0
                * aej_Th
                * (np.exp(lambda_232 * t_old) - np.exp(lambda_232 * t_young)) 
                + aej_Sm
                * (np.exp(lambda_147 * t_old) - np.exp(lambda_147 * t_young))
                )

                #production array
                production = all_alphas * r_positions * beta * allow

                #Neumann inner boundary condition (u[0]_i = -u[1]_i, where "0" is imaginary, "1" represents first real index at diagonal[0])
                diagonal[0] = -3.0 - beta
                d[0] = (3.0 - beta) * u[0] - u[1] - production[0]

                #Dirichlet outer boundary condition, u[nodes+1]_i = u[nodes+1]_i+1, where "nodes+1" is imaginary
                diagonal[-1] = -2.0 - beta
                d[-1] = (2.0 - beta) * u[-1] - u[-2] - production[-1]

                #fill in the rest
                diagonal[1:-1] = -2.0 - beta                   
                d[1:-1] = (2.0 - beta) * u[1:-1] - u[2:] - u[:-2] - production[1:-1]

                #solve it using Thomas algorithm helper function, u becomes u_n+1 and repeat
                u = _tridiag(a, diagonal, c, d, nodes)

        #convert u to the He concentration profile
        He_profile = u / r_positions
        return He_profile

@jit(nopython=True)
def _mp_diffusion_core( 
            nodes, 
            r_step, 
            tT_path, 
            D_sc,
            D_v,
            kappa_1,
            kappa_2,
            f, 
            tolerance, 
            aej_U238, 
            aej_U235, 
            aej_Th, 
            aej_Sm, 
            init_fast_He,
            init_lat_He,
            allow,
            M,
            initial_damp,
            ):
        """ 
        Core function for solving diffusion in a crystal with multiple diffusion pathways, a fast path and a volume path, using a Crank-Nicoloson scheme. Uses jit decorator. Based on the set-up of Lee and Aldama (1992) (https://doi.org/10.1016/0098-3004(92)90093-7) and their algorithm for solving equations 18a and 18b. Adapted by Guenthner et al. (in prep) (doi pending) for a tridiagonal setup, production terms, and Neumann and Dirichlet-type boundary conditions.

        Parameters
        ----------

        nodes: int
            Number of nodes for 1D finite difference diffusion solver

        r_step: float
            Grid spacing in micrometers, reflects 1st node position as 0.5 * r_step from grain center

        tT_path: 2D array of floats
            Time temperature history along which to calculate diffusion

        D_sc: 1D array of floats
            Short-circuit diffusivities. Must be units of micrometer2/s and length of np.size(tT_path,0) - 1.

        D_v: 1D array of floats
            Volume diffusivities. Must be units of micrometer2/s and length of np.size(tT_path,0) - 1.

        kappa_1: float
            Exchange coefficient from 
        
        kappa_2: float
            Exchange coefficient from
        
        f: float
            Fraction of amorphousness

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
        
        init_fast_He: 1D array
            1D profile of alphas for fast path, must be length of nodes. Default is zeros. Must be in terms of radial position (see mp_profile function).
        
        init_lat_He: 1D array
            1D profile of alphas for lattice, must be length of nodes. Default is zeros. Must be in terms of radial position (see mp_profile function). 
        
        allow: float
            Allows for alpha production (equals 1.0) or no alpha production (equals 0.0) during diffusion, useful for generating Arrhenius trends.

        divide: boolean
            Allows for the time-temperature history to be sub-divided for better precision on the concentration profile. Useful for laboratory heating schedules with small fractional losses.

        M: int
            Number of sub-divisions for the exponentially increasing damping function.

        initial_damp: int
            Number of sub-divions for the initial damping.
        
        Returns
        -------

        bulk_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the bulk grain
        
        fast_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the fast pathways

        lat_He_profile: 1D array of floats
            The 1D radial profile of diffused He in the lattice

        """

        #set up arrays for tridiagonal matrix
        #u = vr, v is the He profile, r is radius
        #u_fp is the fast path, and u_lat is the lattice coordinate transform vector
        #u_fp_n and u_lat_n are previous time step vectors 
        
        u_fp_n = init_fast_He.copy()
        u_lat_n = init_lat_He.copy()
    
        #precompute r positions
        r_positions = (np.arange(nodes) + 0.5) * r_step

        #setup diagonal and d (RHS vector, Ax = d)     
        diagonal = np.zeros(nodes)
        d = np.zeros(nodes)
        
        #a is sub-, and c is supra-diagonal
        a = np.ones(nodes)
        a[-1] = 0.0
        c = np.ones(nodes)
        c[0] = 0.0
        
        #step through time from old to young
        for i in range(tT_path.shape[0] - 1):
            
            dt_int = tT_path[i, 0] - tT_path[i + 1, 0]
            fourier_sc = (D_sc[i] * dt_int) / r_step**2

            #subdivide the time step if necessary
            if fourier_sc > 0.5:
                temp = (tT_path[i, 1] + tT_path[i + 1, 1]) / 2
                sub_tT_path = _divide_tT(D_sc[i], dt_int, temp, r_step, M, initial_damp)
            else:
                sub_tT_path = tT_path[i:i + 2, :]

            #perform diffusion at sub-interval time spacing
            for j in range(sub_tT_path.shape[0] - 1):
                t_old = sub_tT_path[j, 0]
                t_young = sub_tT_path[j + 1, 0]
                dt = t_old - t_young 
                beta_sc = (2.0 * r_step**2) / (D_sc[i] * dt)
                beta_v = (2.0 * r_step**2) / (D_v[i] * dt)

                #initial guess at each sub-interval of u_lat (use previous time step)
                u_lat = u_lat_n

                #create alpha production array
                all_alphas = (
                            8
                            * aej_U238
                            * (np.exp(lambda_238 * t_old) - np.exp(lambda_238 * t_young)) 
                            + 7
                            * aej_U235
                            * (np.exp(lambda_235 * t_old) - np.exp(lambda_235 * t_young)) 
                            + 6
                            * aej_Th
                            * (np.exp(lambda_232 * t_old) - np.exp(lambda_232 * t_young)) 
                            + aej_Sm
                            * (np.exp(lambda_147 * t_old) - np.exp(lambda_147 * t_young))
                        )

                #generate RHS vector for fast path concentration from initial u_lat guess
                #Neumann inner BC (u[0]_i = -u[1]_i, where "0" is imaginary, "1" represents first real index at diagonal[0])
                #production term for 1st node
                production = all_alphas[0] * f * 0.5 * r_step * beta_sc * allow
                partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[0]
                partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[0]

                diagonal[0] = -3.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                d[0] = (3.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[0] - u_fp_n[1] - production + partition + partition_n_plus
                
                #Dirichlet outer boundary condition, u[nodes+1]_i = u[nodes+1]_i+1, where "nodes+1" is imaginary
                #production term for last node
                production = all_alphas[-1] * f * (nodes - 0.5) * r_step * beta_sc * allow
                partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[-1]
                partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[-1]
 
                diagonal[-1]= -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                d[-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[-1] - u_fp_n[-2] - production + partition + partition_n_plus
                
                #fill in the rest
                diagonal[1:-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                
                #production term
                production = all_alphas[1:-1] * f * r_positions[1:-1] * beta_sc * allow
                partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[1:-1]
                partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[1:-1]
                
                d[1:-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[1:-1] - u_fp_n[2:] - u_fp_n[:-2] - production + partition + partition_n_plus
                
                #solve for fast path concentration using thomas solve helper function
                u_fp = _tridiag(a, diagonal, c, d, nodes)
                
                #iterate within each time segment for distribution between fast path and lattice
                diff_max = 1.0
                counter = 0
                
                while diff_max > tolerance and counter < 50:

                    #for comparison with updated values in diff_max
                    u_fp_old = u_fp
                    u_lat_old = u_lat
                                     
                    #generate RHS vector for lattice concentration from fast path solution
                    #Neumann inner BC
                    production = all_alphas[0] * (1 - f) * 0.5 * r_step * beta_v * allow
                    partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[0]
                    partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[0]
                    
                    diagonal[0] = -3.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    d[0] = (3.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[0] - u_lat_n[1] - production - partition - partition_n_plus

                    #Dirichlet outer BC
                    production = all_alphas[-1] * (1 - f) * (nodes - 0.5) * r_step * beta_v * allow
                    partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[-1]
                    partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[-1]

                    diagonal[-1] = -2.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    d[-1] = (2.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[-1] - u_lat_n[-2] - production - partition - partition_n_plus
                    
                    #fill in the rest
                    diagonal[1:nodes-1] = -2.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    production = all_alphas[1:-1] * (1 - f) * r_positions[1:-1] * beta_v * allow
                    partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[1:-1]
                    partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[1:-1]
                    d[1:-1] = (2.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[1:-1] - u_lat_n[2:] - u_lat_n[:-2] - production - partition - partition_n_plus
                        
                    #solve for lattice concentration using thomas solve helper function
                    u_lat = _tridiag(a, diagonal, c, d, nodes)
                    
                    #generate RHS vector again for fast path concentration from lattice solution
                    #Neumann inner BC
                    production = all_alphas[0] * f * 0.5 * r_step * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[0]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[0]

                    diagonal[0] = -3.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    d[0] = (3.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[0] - u_fp_n[1] - production + partition + partition_n_plus
                    
                    #Dirichlet outer BC
                    production = all_alphas[-1] * f * (nodes - 0.5) * r_step * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[-1]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[-1]

                    diagonal[-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    d[-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[-1] - u_fp_n[-2] - production + partition + partition_n_plus

                    #fill in the rest
                    diagonal[1:nodes-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    production = all_alphas[1:-1] * f * r_positions[1:-1] * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[1:-1]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[1:-1]

                    d[1:-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[1:-1] - u_fp_n[2:] - u_fp_n[:-2] - production + partition + partition_n_plus

                    #solve for fast path concentration once more using thomas solve helper function
                    u_fp = _tridiag(a, diagonal, c, d, nodes)
                    
                    #determine diff_max to compare for next iteration of while loop
                    xi_max = np.max(np.abs(u_fp - u_fp_old))
                    omega_max = np.max(np.abs(u_lat - u_lat_old))
                    diff_max = max(xi_max, omega_max)
                    counter = counter + 1
                
                #update u_n vectors and move to the next sub-interval time step
                u_fp_n = u_fp
                u_lat_n = u_lat  
        
        #convert each u profile to a He concentration profile
        r_vals = (np.arange(nodes) + 0.5) * r_step
        fast_He_profile = u_fp_n / r_vals
        lat_He_profile  = u_lat_n / r_vals
        bulk_He_profile = (u_fp_n + u_lat_n) / r_vals
        
        return bulk_He_profile, fast_He_profile, lat_He_profile

@jit(nopython=True)
def _teq_rho_r(
            rho_r_array, 
            time, 
            temp, 
            C0, 
            C1, 
            C2, 
            C3, 
            alpha, 
            rmr0, 
            kappa, 
            is_apatite,
            total_anneal,
            relevant_tracks
            ):
    """
    Core function for generating the rho_r array using equivalent time principle. 

    Parameters
    ----------

    rho_r_array: 1D array of floats
        The pre-shortened array of rho_r values, absent values that need to calculated with equivalent time.
    
    time: 1D array of floats
        Pre-shortened array of time values (in seconds)
    
    temp: 1D array of floats
        Pre-shortened array of temp values (in kelvin)
    
    C0: float
        Annealing parameter
    
    C1: float
        Annealing parameter
    
    C2: float
        Annealing parameter
    
    C3: float
        Annealing parameter
    
    alpha: float
        Annealing parameter

    rmr0: float
        Annealing parameter. Only used in apatite annealing. Value of 0.0 should be used if not apatite

    kappa: float
        Annealing parameter. Only used in apatite annealing. Value of 0.0 should be used if not apatite

    is_apatite: boolean
        Apatite check. Use 'True' for apatite annealing
    
    total_anneal: float
        Lower limit on r value that corresponds to a total annealing of a track.

    relevant_tracks: int
        The time-temp position of the oldest remaining track


    Returns
    -------

    rho_r_array: 1D array of floats
        The filled-in rho_r array after applying equivalent time algorithm

    """
    for i in range(relevant_tracks - 2, -1, -1):
        t_eq = 0.0
        temp_mean = np.log(2 / (temp[i] + temp[i + 1]))
        
        for j in range(i, -1, -1):
            time_step = time[j] - time[j + 1] + t_eq
            
            #just in case there's a zero time step
            if time_step <= 0:
                continue
            
            r = (C0 + C1 * ((np.log(time_step) - C2) / (temp_mean - C3))) ** (1 / alpha) + 1
            r = 0.0 if r <= 0 else 1 / r

            #convert to reduced density (mineral type dependent)
            if r <= total_anneal:
                rho_r_array[i, j] = 0.0
                break
            else:
                if is_apatite:
                    rc_lr = ((r - rmr0) / (1 - rmr0)) ** kappa
                    if rc_lr >= 0.765:
                        rho_r_array[i, j] = 1.6 * rc_lr - 0.6 
                    else: 
                        rho_r_array[i, j] = 9.205 * rc_lr**2 - 9.157 * rc_lr + 2.269
                else:
                    rho_r_array[i, j] = 1.25 * (r - 0.2)
            
            #calculate t_eq, prevent subzero indexing
            if j == 0:
                break
            elif r < 1:
                temp_mean = np.log(2 / (temp[j - 1] + temp[j]))
                t_eq = np.exp(C2 + (temp_mean - C3) * (((1 / r) - 1) ** alpha - C0) / C1)
    
    return rho_r_array