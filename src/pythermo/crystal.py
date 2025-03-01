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
from scipy.linalg import solve_banded
from scipy.integrate import romb

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

        aej_U238, aej_U235, aej_Th, aej_Sm: lists of floats
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

        #alpha ejection profile lists for each isotope
        aej_U238 = [
            (
                U238_atom 
                * (
                    0.5 
                    + (
                        (
                            (((i + 0.5) * r_step) ** 2 + radius**2 - as_U238**2)
                            / (2 * ((i + 0.5) * r_step))
                        ) 
                        - ((i + 0.5) * r_step)
                    )
                    / (2 * as_U238)
                )
                if ((i + 0.5) * r_step) >= (radius - as_U238) 
                else U238_atom
            )
            for i in range(nodes)
        ]
        
        aej_U235 = [
            (
                U235_atom 
                * (
                    0.5 
                    + (
                        (
                            (((i + 0.5) * r_step) ** 2 + radius**2 - as_U235**2)
                            / (2 * ((i + 0.5) * r_step))
                        ) 
                        - ((i + 0.5) * r_step)
                    )
                    / (2 * as_U235)
                ) 
                if ((i + 0.5) * r_step) >= (radius - as_U235) 
                else U235_atom
            ) 
            for i in range(nodes)
        ]
        
        if Th_atom == 0:
            aej_Th = [0 for i in range(nodes)]
        else:
            aej_Th = [
                (
                    Th_atom 
                    * (
                        0.5 
                        + (
                            (
                                (((i + 0.5) * r_step) ** 2 + radius**2 - as_Th**2)
                                / (2 * ((i + 0.5) * r_step))
                            ) 
                            - ((i + 0.5) * r_step)
                        )
                        / (2 * as_Th)
                    ) 
                    if ((i + 0.5) * r_step) >= (radius - as_Th) 
                    else Th_atom
                ) 
                for i in range(nodes)
            ]
        
        if Sm_atom == 0:
            aej_Sm = [0 for i in range(nodes)]
        else:
            aej_Sm = [
                (
                    Sm_atom 
                    * (
                        0.5 
                        + (
                            (
                                (((i + 0.5) * r_step) ** 2 + radius**2 - as_Sm**2)
                                / (2 * ((i + 0.5) * r_step))
                            ) 
                            - ((i + 0.5) * r_step)
                        )
                        / (2 * as_Sm)
                    ) 
                    if ((i + 0.5) * r_step) >= (radius - as_Sm) 
                    else Sm_atom
                ) 
                for i in range(nodes)
            ]
        
        #calculate alpha ejection correction factors for age equation, use an accumulator approach
        total_U238 = 0.0
        total_U235 = 0.0
        total_Th = 0.0
        total_Sm = 0.0

        nft_U238 = 0.0
        nft_U235 = 0.0
        nft_Th = 0.0
        nft_Sm = 0.0

        dft_U238 = 0.0
        dft_U235 = 0.0
        dft_Th = 0.0
        dft_Sm = 0.0

        inner_vol = 0.0
        rad_position = 0.0
        for i in range(nodes):
            rad_position = rad_position + r_step
            outer_vol = rad_position**3
            vol = outer_vol - inner_vol

            total_U238 = total_U238 + U238_atom * (outer_vol - inner_vol)
            nft_U238 = nft_U238 + vol * aej_U238[i]
            dft_U238 = dft_U238 + vol * U238_atom
            total_U235 = total_U235 + U235_atom * (outer_vol - inner_vol)
            nft_U235 = nft_U235 + vol * aej_U235[i]
            dft_U235 = dft_U235 + vol * U235_atom

            if Th_atom == 0:
                #prevents divide by zero
                dft_Th = 1.0
            else:
                total_Th = total_Th + Th_atom * (outer_vol - inner_vol)
                nft_Th = nft_Th + vol * aej_Th[i]
                dft_Th = dft_Th + vol * Th_atom

            if Sm_atom == 0:
                #prevents divide by zero
                dft_Sm = 1.0
            else:
                total_Sm = total_Sm + Sm_atom * (outer_vol - inner_vol)
                nft_Sm = nft_Sm + vol * aej_Sm[i]
                dft_Sm = dft_Sm + vol * Sm_atom

            inner_vol = outer_vol

        #scale by production and volume with a base of 1/(4/3 * PI)
        total_U238 = total_U238 * 8
        total_U235 = total_U235 * 7
        total_Th = total_Th * 6

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
            self, nodes, r_step, tT_path, diffs, aej_U238, aej_U235, aej_Th, aej_Sm
            ):
        """ 
        Solves the diffusion equation with production along a 1D radial profile using the Crank-Nicoloson finite difference scheme. Assumes no flux across the center, or inner boundary, node (Neumann boundary condition) and zero concentration along the outer boundary (Dirichlet boundary condition). Production can be zeroed out by passing arrays of zeros for each of the alphe ejection arrays. Original solution and set-up for the algorithm comes from Ketcham (2005) (https://doi.org/10.2138/rmg.2005.58.11), specifically equation 21, and equations 22 and 26 for boundary conditions.
        
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

        Returns
        -------

        He_profile: list of floats
            The 1D radial profile of diffused He

        """
        #set up arrays for tridiagonal matrix
        #u is the coordinate transform vector
        #u = vr, v is the He profile, r is radius 
        u = np.zeros(nodes)

        #setup diagonal and d (RHS vector, Ax = d)
        #a is sub-, and c is supra-diagonal
        
        diagonal = np.zeros(nodes)
        d = np.zeros(nodes)
        
        #a is sub-, and c is supra-diagonal (tridiagonal matrix solver, for instructional purposes only)
        #a = np.ones(nodes)
        #c = np.ones(nodes-1)

        #a is sub-, and c is supra-diagonal (scipy banded matrix solver)
        a = np.ones(nodes)
        a[-1] = 0
        c = np.ones(nodes)
        c[0] = 0

        #step through time from old to young
        for i in range(np.size(tT_path, 0) - 1):
            t_old = tT_path[i, 0]
            t_young = tT_path[i + 1, 0]
            dt = t_old - t_young
            
            beta = (2.0 * r_step**2) / (diffs[i] * dt) 

            #Neumann inner boundary condition (u[0]_i = -u[1]_i, where "0" is imaginary, "1" represents first real index at diagonal[0])
            #production term for 1st node
            alphas = (
                8
                * aej_U238[0]
                * (np.exp(lambda_238 * t_old) - np.exp(lambda_238 * t_young)) 
                + 7
                * aej_U235[0]
                * (np.exp(lambda_235 * t_old) - np.exp(lambda_235 * t_young)) 
                + 6
                * aej_Th[0]
                * (np.exp(lambda_232 * t_old) - np.exp(lambda_232 * t_young)) 
                + aej_Sm[0]
                * (np.exp(lambda_147 * t_old) - np.exp(lambda_147 * t_young))
            )
            production = alphas * 0.5 * r_step * beta

            diagonal[0] = -3.0 - beta
            d[0] = (3.0 - beta) * u[0] - u[1] - production

            #Dirichlet outer boundary condition, u[nodes+1]_i = u[nodes+1]_i+1, where "nodes+1" is imaginary
            #production term for last node
            alphas = (
                8
                * aej_U238[-1]
                * (np.exp(lambda_238 * t_old) - np.exp(lambda_238 * t_young)) 
                + 7
                * aej_U235[-1]
                * (np.exp(lambda_235 * t_old) - np.exp(lambda_235 * t_young)) 
                + 6
                * aej_Th[-1]
                * (np.exp(lambda_232 * t_old) - np.exp(lambda_232 * t_young)) 
                + aej_Sm[-1]
                * (np.exp(lambda_147 * t_old) - np.exp(lambda_147 * t_young))
            )
            production = alphas * (nodes - 0.5) * r_step * beta

            diagonal[-1] = -2.0 - beta
            d[-1] = (2.0 - beta) * u[-1] - u[-2] - production

            #fill in the rest
            diagonal[1:nodes-1] = -2.0 - beta
            for j in range(1, nodes - 1):
                #production term 
                alphas = (
                    8
                    * aej_U238[j]
                    * (np.exp(lambda_238 * t_old) - np.exp(lambda_238 * t_young)) 
                    + 7
                    * aej_U235[j]
                    * (np.exp(lambda_235 * t_old) - np.exp(lambda_235 * t_young)) 
                    + 6
                    * aej_Th[j]
                    * (np.exp(lambda_232 * t_old) - np.exp(lambda_232 * t_young)) 
                    + aej_Sm[j]
                    * (np.exp(lambda_147 * t_old) - np.exp(lambda_147 * t_young))
                )
                production = alphas * (j + 0.5) * r_step * beta
                
                d[j] = (2.0 - beta) * u[j] - u[j+1] - u[j-1] - production
            
            #solve it using tridiaonal matrix algorithm, for instructional purposes only, u becomes u_n+1 and repeat
            #u = self.tridiag(a, diagonal, c, d, nodes)
            
            #solve it using scipy banded solver, u becomes u_n+1 and repeat
            A = [c, diagonal, a]
            u = solve_banded((1, 1), A, d)

        #convert u to the He concentration profile
        He_profile = [u[i] / ((i + 0.5) * r_step) for i in range(0, nodes)]
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
        Solves for diffusion in a crystal with multiple diffusion pathways, a fast path and a volume path, using a Crank-Nicoloson scheme. Based on the set-up of Lee and Aldama (1992) (https://doi.org/10.1016/0098-3004(92)90093-7) and their algorithm for solving equations 18a and 18b. Uses equatios from Ketcham (2005) (https://doi.org/10.2138/rmg.2005.58.11), specifically equation 21 for tridiagonal setup, and equations 22 and 26 for boundary conditions (which are Neumann and Dirichlet-type in both publications).

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

        bulk_He_profile: list of floats
            The 1D radial profile of diffused He in the bulk grain
        
        fast_He_profile: list of floats
            The 1D radial profile of diffused He in the fast pathways

        lat_He_profile: list of floats
            The 1D radial profile of diffused He in the lattice

        """
        #unpack diff_parameters, D values have to be in units of micrometer**2/s
        D_sc = diff_parameters['D_sc']
        D_v = diff_parameters['D_v']
        kappa_1 = diff_parameters['kappa_1']
        kappa_2 = diff_parameters['kappa_2']
        f = diff_parameters['f']

        #lambda function for alpha production calculation
        decay = lambda rad_pos, t_1, t_2: (
                8
                * aej_U238[rad_pos]
                * (np.exp(lambda_238 * t_1) - np.exp(lambda_238 * t_2)) 
                + 7
                * aej_U235[rad_pos]
                * (np.exp(lambda_235 * t_1) - np.exp(lambda_235 * t_2)) 
                + 6
                * aej_Th[rad_pos]
                * (np.exp(lambda_232 * t_1) - np.exp(lambda_232 * t_2)) 
                + aej_Sm[rad_pos]
                * (np.exp(lambda_147 * t_1) - np.exp(lambda_147 * t_2))
            )

        #set up arrays for tridiagonal matrix
        #u = vr, v is the He profile, r is radius
        #u_fp is the fast path, and u_lat is the lattice coordinate transform vector
        #u_fp_n and u_lat_n are previous time step vectors 

        if init_fast_He is not None and init_lat_He is not None:
            u_fp_n = init_fast_He
            u_lat_n = init_lat_He
        else:
            u_fp_n = np.zeros(nodes)
            u_lat_n = np.zeros(nodes)

        #setup diagonal and d (RHS vector, Ax = d)     
        diagonal = np.zeros(nodes)
        d = np.zeros(nodes)
        
        #a is sub-, and c is supra-diagonal
        a = np.ones(nodes)
        a[-1] = 0
        c = np.ones(nodes)
        c[0] = 0

        #determine if production occurs within each time step
        if produce:
            allow = 1
        else:
            allow = 0
        
        #step through time from old to young
        for i in range(np.size(tT_path, 0) - 1):
            
            dt_int = tT_path[i, 0] - tT_path[i + 1, 0]
            fourier_sc = (D_sc[i] * dt_int) / r_step**2

            #subdivide the time step if necessary
            if fourier_sc > 0.5:
                temp = (tT_path[i, 1] + tT_path[i + 1, 1]) / 2
                M = 100
                initial_damp = 20
                sub_tT_path = self.divide_tT(D_sc[i], dt_int, temp, r_step, M, initial_damp)
            else:
                sub_tT_path = tT_path[i:i + 1, :]

            #perform diffusion at sub-interval time spacing
            for j in range(np.size(sub_tT_path, 0) - 1):
                t_old = sub_tT_path[j, 0]
                t_young = sub_tT_path[j + 1, 0]
                dt = t_old - t_young 
                beta_sc = (2.0 * r_step**2) / (D_sc[i] * dt)
                beta_v = (2.0 * r_step**2) / (D_v[i] * dt)

                #initial guess at each sub-interval of u_lat (use previous time step)
                u_lat = u_lat_n

                #generate RHS vector for fast path concentration from initial u_lat guess
                #Neumann inner BC (u[0]_i = -u[1]_i, where "0" is imaginary, "1" represents first real index at diagonal[0])
                #production term for 1st node
                alphas = decay(0, t_old, t_young)
                production = alphas * f * 0.5 * r_step * beta_sc * allow
                partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[0]
                partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[0]

                diagonal[0] = -3.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                d[0] = (3.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[0] - u_fp_n[1] - production + partition + partition_n_plus
                
                #Dirichlet outer boundary condition, u[nodes+1]_i = u[nodes+1]_i+1, where "nodes+1" is imaginary
                #production term for last node
                alphas = decay(-1, t_old, t_young)
                production = alphas * f * (nodes - 0.5) * r_step * beta_sc * allow
                partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[-1]
                partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[-1]
 
                diagonal[-1]= -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                d[-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[-1] - u_fp_n[-2] - production + partition + partition_n_plus
                
                #fill in the rest
                diagonal[1:nodes-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                for k in range(1, nodes - 1):
                    #production term 
                    alphas = decay(k, t_old, t_young)
                    production = alphas * f * (k + 0.5) * r_step * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[k]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[k]
                    
                    d[k] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[k] - u_fp_n[k+1] - u_fp_n[k-1] - production + partition + partition_n_plus
                
                #solve for fast path concentration using scipy banded solver
                A = [c, diagonal, a]
                u_fp = solve_banded((1, 1), A, d)
                
                #iterate within each time segment for distribution between fast path and lattice
                diff_max = 1.0
                counter = 0
                
                while diff_max > tolerance and counter < 50:

                    #for comparison with updated values in diff_max
                    u_fp_old = u_fp
                    u_lat_old = u_lat
                                     
                    #generate RHS vector for lattice concentration from fast path solution
                    #Neumann inner BC
                    alphas = decay(0, t_old, t_young)
                    production = alphas * (1 - f) * 0.5 * r_step * beta_v * allow
                    partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[0]
                    partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[0]
                    
                    diagonal[0] = -3.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    d[0] = (3.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[0] - u_lat_n[1] - production - partition - partition_n_plus

                    #Dirichlet outer BC
                    alphas = decay(-1, t_old, t_young)
                    production = alphas * (1 - f) * (nodes - 0.5) * r_step * beta_v * allow
                    partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[-1]
                    partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[-1]

                    diagonal[-1] = -2.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    d[-1] = (2.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[-1] - u_lat_n[-2] - production - partition - partition_n_plus
                    
                    #fill in the rest
                    diagonal[1:nodes-1] = -2.0 - beta_v + beta_v * 0.5 * dt * kappa_2
                    for k in range(1, nodes - 1):
                        alphas = decay(k, t_old, t_young)
                        production = alphas * (1 - f) * (k + 0.5) * r_step * beta_v * allow
                        partition = beta_v * 0.5 * dt * kappa_1 * u_fp_n[k]
                        partition_n_plus = beta_v * 0.5 * dt * kappa_1 * u_fp[k]
                        d[k] = (2.0 - beta_v - beta_v * 0.5 * dt * kappa_2) * u_lat_n[k] - u_lat_n[k+1] - u_lat_n[k-1] - production - partition - partition_n_plus
                        

                    #solve for lattice concentration using scipy banded solver
                    A = [c, diagonal, a]
                    u_lat = solve_banded((1, 1), A, d)
                    
                    #generate RHS vector again for fast path concentration from lattice solution
                    #Neumann inner BC
                    alphas = decay(0, t_old, t_young)
                    production = alphas * f * 0.5 * r_step * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[0]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[0]

                    diagonal[0] = -3.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    d[0] = (3.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[0] - u_fp_n[1] - production + partition + partition_n_plus
                    
                    #Dirichlet outer BC
                    alphas = decay(-1, t_old, t_young)
                    production = alphas * f * (nodes - 0.5) * r_step * beta_sc * allow
                    partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[-1]
                    partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[-1]

                    diagonal[-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    d[-1] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[-1] - u_fp_n[-2] - production + partition + partition_n_plus

                    #fill in the rest
                    diagonal[1:nodes-1] = -2.0 - beta_sc - beta_sc * 0.5 * dt * kappa_1
                    for k in range(1, nodes - 1):
                        alphas = decay(k, t_old, t_young)
                        production = alphas * f * (k + 0.5) * r_step * beta_sc * allow
                        partition = beta_sc * 0.5 * dt * kappa_2 * u_lat_n[k]
                        partition_n_plus = beta_sc * 0.5 * dt * kappa_2 * u_lat[k]

                        d[k] = (2.0 - beta_sc + beta_sc * 0.5 * dt * kappa_1) * u_fp_n[k] - u_fp_n[k+1] - u_fp_n[k-1] - production + partition + partition_n_plus

                    #solve for fast path concentration once more using scipy banded solver
                    A = [c, diagonal, a]
                    u_fp = solve_banded((1, 1), A, d)
                    
                    #determine diff_max to compare for next iteration of while loop
                    xi_max = np.max(np.abs(u_fp - u_fp_old))
                    omega_max = np.max(np.abs(u_lat - u_lat_old))
                    diff_max = max(xi_max, omega_max)
                    counter = counter + 1
                
                #update u_n vectors and move to the next sub-interval time step
                u_fp_n = u_fp
                u_lat_n = u_lat  
        
        #convert each u profile to a He concentration profile
        fast_He_profile = [u_fp_n[i] / ((i + 0.5) * r_step) for i in range(0, nodes)]
        lat_He_profile = [u_lat_n[i] / ((i + 0.5) * r_step) for i in range(0, nodes)]
        bulk_He_profile = [(u_fp_n[i] + u_lat_n[i])/ ((i + 0.5) * r_step) for i in range(0, nodes)]
        
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
        #set up iterative Newton-Raphson solver for eq 24 from Britz et al. (2003)
        fourier_calc = D * dt / r_step**2
        beta_guess = 1.2
        f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
        f_beta_prime = 2*fourier_calc*beta_guess - M*beta_guess**(M-1)
        tolerance = 1e-6

        beta_diff = beta_guess

        #iterate to solve
        while abs(beta_diff) > tolerance:
            beta = beta_guess - f_beta / f_beta_prime
            beta_diff = beta_guess - beta
            beta_guess = beta
            f_beta = fourier_calc * (beta_guess**2 - 1) - (beta_guess**M - 1)
            f_beta_prime = 2 * fourier_calc * beta_guess - M * beta_guess**(M - 1)

        #add on the initial damping steps
        fourier_set = 0.5
        dt_int = fourier_set * r_step**2 / D
        expansion_steps = initial_damp + M

        sub_tT = np.zeros((expansion_steps,2))
        sub_tT[0,0] = dt
        for i in range(1, initial_damp):
            sub_tT[i,0] = dt -  (dt_int + i * dt_int)

        tau_1 = sub_tT[i,0] * (beta - 1) / (beta**M - 1)

        delta_t = 0
        for i in range(M-1):
            delta_t = delta_t + beta**i * tau_1
            sub_tT[initial_damp + i,0] = sub_tT[initial_damp-1,0] - delta_t

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

    def tridiag(self, a, diagonal, c, d, nodes):
        """ 
        Implements the tridiagonal matrix algorithm for the CN diffusion solver. Not called in CN diffusion. Here for instructional purposes. Adapted for python from Numerical Recipes, section 2.4, Press et al. (2007) ISBN: 978-0-521-88068-8

        Parameters
        ----------

        a: 1D array of floats
            Values along the subdiagonal of the tridiagonal matrix

        diagonal: 1D array of floats
            Values along the diagonal of the tridiagonal matrix
        
        c: 1D array of floats
            Values along the supradiagonal of the tridiagonal matrix

        d: 1D array of floats
            Values on the right-hand side of equation

        nodes: int
            Number of nodes in the diffusion solver (tridiagonal matrix is n x n where n = nodes)

        Returns
        -------

        x: 1D array of floats
            The inverted matrix (vector)

        """

        #gam and x 
        gam = np.zeros(nodes)
        x = np.zeros(nodes)
        
        #perform forward sweep
        bet = diagonal[0]
        x[0] = d[0] / bet

        for i in range(1, nodes):
            gam[i] = c[i - 1] / bet
            bet = diagonal[i] - a[i] * gam[i]
            x[i] = (d[i] - a[i] * x[i - 1]) / bet 

        #back substitute
        for j in range(nodes - 2, -1, -1):
            x[j] = x[j] - gam[j + 1] * x[j + 1]

        return x

    def romberg(self, integral, a, b, log2_nodes, r_step):
        """ 
        Implements Romberg's method for using the extended trapezoidal rule. Not called in CN diffusion. Here for instructional purposes. Adapated from Numerical Recipes, sections 4.2 and 4.3, Press et al. (2007) ISBN: 978-0-521-88068-8. Works on a vector of 2**n + 1 equally spaced samples of a function.

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
        Polynomial interpolation method (Neville's algorithm), adapted from Numerical Recipes section 3.2, Press et al. (2007) ISBN: 978-0-521-88068-8. Given a starting value, x, two vectors of x values and f(x) solutions, use polynomials up to order jl-1 to interpolate and accumulate the rest of the y values. Not called in CN diffusion. Here for instructional purposes.

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
            
    def guenthner_date(self):
        """
        Zircon (U-Th)/He date calculator. First, calculates the diffusivity at each time step of class variable relevant_tT using the parameterization of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally, the parent class method He_date is called to convert the He profile to a (U-Th)/He date.
        
        Returns
        -------
        
        date: float
            Zircon (U-Th)/He date, corrected for alpha ejection
        
        """
        #calculate damage levels using guenthner_damage function
        damage = self.guenthner_damage()
        
        #get diff_list from guenthner_diffs method
        diff_list = self.guenthner_diffs(damage)

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
            return total_He

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
        
        return date
    
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
            Array of the bulk diffusivities as a function of damage at each time step of relevant_tT. Length is one less than the number of rows in relevant_tT (because last time step is 0). Diffusivities are in 1/s.
        
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
        D0_z = 2.0e-3 * 10**8 
        D0_N17 = 0.0034 * 10**8
        Ea_z = 52.0
        Ea_N17 = 71.0
        Ea_trap = 160 
        gamma = 2e-19
        omega = 0.25
        kappa_2 = -1e-4
        k_star = 1e-14
        n = 10

        for i in range(len(damage)):
            #average temp between time steps in K
            temp_K = ((relevant_tT[i, 1] + 273.15) + (relevant_tT[i + 1, 1] + 273.15)) / 2

            #tort parameters
            f_a_DI = 1 - np.exp(-B_a * damage[i])
            l_int = (4.2 / (f_a_DI * SV)) - 2.5
            tau = (l_int_0 / l_int)**2
            D0_v = D0_z * (1 / tau)

            #trap parameters
            psi = (gamma * damage[i]**omega * np.exp(Ea_trap/(gas_constant * temp_K))) + 1

            #lattice (volume) diffusivity
            D_v = D0_v * np.exp(-Ea_z / (gas_constant * temp_K)) / psi
            lat_diff_array[i] = D_v

            #fast path (short-circuit) diffusivity
            D_sc = D0_N17 * np.exp(-Ea_N17 / (gas_constant * temp_K))
            fast_diff_array[i] = D_sc
            
            #fraction amorphous
            f = 1 - ((1 + B_a * damage[i]) * np.exp(-(B_a * damage[i])))**n

            #treat kappa_2 as a constant, solve for kappa_1
            kappa_1 = -kappa_2 * (1 - f) / (k_star * f)

            #bulk diffusivity
            D_b_a2 = (
                (1 / (kappa_1 - kappa_2)) * 
                (-kappa_2 * D_sc / radius**2 + kappa_1 * D_v / radius**2)
            )
            bulk_diff_array[i] = D_b_a2
        
        return fast_diff_array, lat_diff_array, bulk_diff_array
    
    def mp_profile(
            self, 
            diff_parameters, 
            tolerance, 
            init_fast_He=None, 
            init_lat_He=None, 
            eject=True, 
            produce=True,
            ):
        """
        Returns the 1D, spherical diffusion profiles for lattice, fast path, and the bulk grain using the multi-path diffusion function. Used for Arrhenius plotting.

        Parameters
        ----------

        diff_parameters: dictionary of floats
            Fitted parameters for multi-path diffusion. Diffusivities must have units of microns^2/s

        tolerance: float
            Convergence criterion for iterative diffusion algorithm

        init_fast_He: optional 1D array
            1D profile of alphas (in atoms/g) for fast path, must be length of nodes. Default is None.
        
        init_lat_He: optional 1D array
            1D profile of alphas (in atoms/g) for lattice, must be length of nodes. Default is None.
        
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

        total_He: float
            The total amount of helium present in atoms per spherical volume (base of 1/(4/3 * Pi))
 
        
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

        bulk_He, fast_He, lat_He = self.mp_diffusion(
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

        #use Romberg integration to calculate total amount of He
        #check for zero helium concentration
        minimum_He = bulk_He[0] * 0.5 * self.__r_step
        for i in range(self.__nodes):
            if bulk_He[i] < minimum_He:
                minimum_He = bulk_He[i]
        
        if minimum_He < -bulk_He[0] * 0.5:
            total_He = 0
            return total_He

        #convert He profile into a spherical function for integration
        integral = [
            bulk_He[i] * 4 * np.pi * ((0.5 + i) * self.__r_step) ** 2 
            for i in range(self.__nodes)
        ]

        total_He = romb(integral, self.__r_step)

        #units in atoms per volume (base of 1/(4/3 * Pi))
        total_He = total_He / ((4 / 3) * np.pi)

        return bulk_He, fast_He, lat_He, total_He

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
    
    def flowers_diffs(self,damage):
        """
        Calculates the diffusivity at each time step of class variable relevant_tT using the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015).

        Parameters
        ----------

        damage: 1D array of floats
            Array of total amount of damage at each time step of relevant_tT

        
        Returns
        -------
        
        diff_list: list of floats
            List of the diffusivities as a function of damage at each time step of relevant_tT
        
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
    
    def flowers_date(self):
        """
        Apatite (U-Th)/He date calculator. First, calculates the diffusivity at each time step of class variable relevant_tT using the parameterization of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). The diffusivities are then passed to the parent class method CN_diffusion, along with relevant parameters. Finally,the parent class method He_date is called to convert the He profile to a (U-Th)/He date.

        Returns
        -------
        
        date: float
            Apatite (U-Th)/He date, corrected for alpha ejection
        
        """
        #calculate damage levels using flowers_damage function
        damage = self.flowers_damage()

        #calculate diffusivities using flowers_diffs function
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
            return total_He
        
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

        return date
