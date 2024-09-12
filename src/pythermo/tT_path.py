"""
tT_path.py

Contains one class 'tT_path' with methods for interpolating and discretizing time-temperature paths from a select number of time-temperature points. Methods also calculate reduced fission track lengt distributions using an equivalent time approach.

"""
from .constants import np, sec_per_myr

class tT_path:
    def __init__(self, tTin, temp_precision=5, rate_acceleration=1.5):
        """
        Constructor for class tT_path. Assumes input tT path goes from young at index 0 -> old at index -1.

        Parameters
        ----------
        
        tTin: 2D array
            Instance variable, tT points to be interpolated. 1st row is time (in m.y.), 2nd row is temp (in oC)

        temp_precision: optional float
            Default minimal spacing of temperature steps. Smaller numbers yield better precision in the diffusion solver (and date) at the cost of increased run-time. Default is 5.

        rate_acceleration: optional float
            Default maximum temperature acceleration allowed between time points. Smaller numbers yield better precision in the diffusion solver (and date) at the cost of increased run-time Default is 1.5.

        """
        self.__tTin = tTin

        #in degrees C
        self.__precision = temp_precision

        #in millions of years
        self.__acceleration = rate_acceleration

    def get_tTin(self):
        return self.__tTin
    
    def get_precision(self):
        return self.__precision
    
    def get_acceleration(self):
        return self.__acceleration

    def tT_interpolate(self):
        """
        Works on an instance of a tT path with a limited number of points, and expands to include many more points for damage and diffusion modeling. The numpy method "interp" does the bulk of the work here, but some extra code ensures that adequate precision is used for time steps that cover rapid temperature changes. This prevents imprecision in the diffusion code. These values (rate of temperature acceleration, max size of temperature step) have defaults, but can be user defined with each class instance.

        Returns
        -------
        
        tTout: 2D array
            Interpolated array of tT points

        """
        
        #default time step is set to 1% of total length of time
        start_time = self.__tTin[-1,0]
        default_time_step = 0.1 * start_time
        previous_time_step = default_time_step

        #list of time values over which to interpolate, appended to in for loop
        time_list = []

        #for loop steps through each segment of tTin
        for i in range(np.size(self.__tTin, 0) - 1, 0, -1):
            #rate of temperature change (oC/m.y.)
            rate=(self.__tTin[i,1] - self.__tTin[i - 1, 1]) / (
                self.__tTin[i, 0] - self.__tTin[i - 1, 0]
            )
            abs_rate = abs(rate)
            temp_per_time_step = abs_rate * default_time_step
            
            if temp_per_time_step <= self.__precision:
                current_default_time_step = default_time_step
            else:
                current_default_time_step = self.__precision/abs_rate

            time_step = current_default_time_step
            if time_step > previous_time_step * self.__acceleration:
                time_step = previous_time_step * self.__acceleration
            
            #conversion of number of steps to an int rounds down
            #results in some cases to a relatively small amount of round off error 
            number_of_steps = int(
                (self.__tTin[i, 0] - self.__tTin[i - 1, 0]) / time_step
            )
            
            #some error proofing included just in case
            time_segment = [
                self.__tTin[i,0] - time_step * j 
                for j in range(number_of_steps) 
                if self.__tTin[i,0] - time_step*j > self.__tTin[i - 1, 0]
            ]
            
            previous_time_step = time_step
            time_list = time_list + time_segment

        #add on the last time step (present day) and interpolate
        time_list.append(0)
        temp_out = np.interp(time_list, self.__tTin[:, 0], self.__tTin[:, 1])

        #convert time to secs and temperature to Kelvin
        time_array = np.array(time_list) * sec_per_myr
        temp_out = temp_out + 273.15
        tT_out = np.array([time_array, temp_out])

        return tT_out.transpose()

    def anneal(self, kinetics):
        """
        Damage annealing method using the equivalent time concept. For a given tT path, this method determines how far back in time information about annealing is retained and therefore relevant for diffusion and (U-Th)/He date calculation. The initial search algorithm for the oldest relevant track is based on Rich Ketcham's RDAAM code.
        
        Parameters
        ----------
        
        kinetics: list of floats
            Mineral and fission track annealing model specific kinetics, assumes a fanning curvilinear fit was used in Arrhenius space, also include rmr0 and kappa if apatite

        Returns
        -------

        rho_r_array: 2D array of floats
            Reduced track densities for each time slice of an instance of the class' tT path.

        interp_tT: 2D array of floats
            Interpolated and condensed time-temperature path

        """

        #unpack kinetics list
        C0 = kinetics[0]
        C1 = kinetics[1]
        C2 = kinetics[2]
        C3 = kinetics[3]
        alpha = kinetics[4]
        total_anneal = kinetics[5]
        rmr0 = kinetics[6]
        kappa = kinetics[7]

        if rmr0 is None:
            mineral_type = 'zircon'
        else:
            mineral_type = 'apatite'

        #interpolate tT path
        interp_tT = self.tT_interpolate()
        tT_len = np.size(interp_tT, 0)

        #find the oldest track that still exists at the present day, store its annealing history in a list
        t_eq = 0.0
        temp_mean = np.log(2 / (interp_tT[tT_len - 2, 1] + interp_tT[tT_len - 1, 1]))
        
        #youngest track is unannealed
        oldest_track_list = [1]
        
        for i in range(tT_len - 2, -1, -1):
            old_track_time = interp_tT[i, 0] - interp_tT[i + 1, 0] + t_eq

            #just in case there's a zero time step
            if old_track_time <= 0: 
                continue

            r = (
                C0 + C1 * ((np.log(old_track_time) - C2) / (temp_mean - C3))
                ) ** (1 / alpha) + 1

            if r<=0:
                r = 0.0
            else:
                r = 1/r
            
            oldest_track_list.insert(0, r)

            #are we at the oldest? 
            if r <= total_anneal:
                oldest_track_list[0] = 0
                present_old_track = i
                break
            #convert to reduced density (mineral type dependent)
            else:
                if mineral_type == 'apatite':
                    rc_lr = ((r - rmr0)/(1 - rmr0)) ** kappa
                    if rc_lr >= 0.765:
                        oldest_track_list[0] = 1.6 * rc_lr - 0.6
                    else:
                        oldest_track_list[0] = 9.205 * rc_lr**2 - 9.157 * rc_lr + 2.269
                        
                elif mineral_type == 'zircon':
                    oldest_track_list[0] = 1.25 * (r - 0.2)
            
            #calculate t_eq, prevent subzero indexing
            if i == 0:
                present_old_track = 0
                break
            elif r<1:
                temp_mean = np.log(2 / (interp_tT[i - 1, 1] + interp_tT[i, 1]))
                t_eq = np.exp(
                    C2 + (temp_mean - C3) * (((1 / r) - 1) ** alpha - C0) / C1
                )

        #now find the oldest track that still existed when the present day oldest track formed
        if present_old_track > 0:
            t_eq = 0.0
            for i in range(present_old_track - 1, -1, -1):
                oldest_track_time = interp_tT[i, 0] - interp_tT[i + 1, 0] + t_eq

                #just in case there's a zero time step
                if oldest_track_time <= 0: 
                    continue

                r = (
                    C0 + C1 * ((np.log(oldest_track_time) - C2) / (temp_mean - C3))
                    ) ** (1 / alpha) + 1

                if r <= 0:
                    r = 0.0
                else:
                    r = 1 / r

                #are we at the oldest?
                if r <= total_anneal:
                    oldest_track = i
                    break
                
                #calculate t_eq, prevent subzero indexing
                if i == 0:
                    oldest_track = 0
                    break
                elif r < 1:
                    temp_mean = np.log(2 / (interp_tT[i - 1, 1] + interp_tT[i, 1]))
                    t_eq = np.exp(
                        C2 + (temp_mean - C3) * (((1 / r) - 1) ** alpha - C0) / C1
                    )

        else:
            oldest_track = 0

        #calculate the reduced length 2D array, only go back to the oldest track
        relevant_tracks = tT_len - oldest_track
        rho_r_array = np.zeros((relevant_tracks, relevant_tracks))
        
        #shorten the interpolated tT path
        interp_tT = interp_tT[oldest_track:tT_len]

        #fill in youngest row of tracks (history of track that lives to present day)
        rho_r_array[
            relevant_tracks - 1, present_old_track - oldest_track : relevant_tracks
        ] = oldest_track_list

        #fill in the rest
        for i in range(relevant_tracks - 2, -1, -1):
            t_eq = 0.0
            temp_mean = np.log(2 / (interp_tT[i, 1] + interp_tT[i + 1, 1]))
            
            for j in range(i, -1, -1):
                time_step = interp_tT[j, 0] - interp_tT[j + 1, 0] + t_eq
                
                #just in case there's a zero time step
                if time_step <= 0: 
                    continue
                
                r = (
                    C0 + C1 * ((np.log(time_step) - C2) / (temp_mean - C3))
                ) ** (1 / alpha) + 1
                
                if r <= 0:
                    r = 0.0
                else:
                    r = 1 / r

                #convert to reduced density (mineral type dependent)
                if r <= total_anneal:
                    rho_r_array[i, j] = 0
                    break
                else:
                    if mineral_type == 'apatite':
                        rc_lr = ((r - rmr0) / (1 - rmr0)) ** kappa
                        if rc_lr >= 0.765:
                            rho_r_array[i, j] = 1.6 * rc_lr -0.6
                        else:
                            rho_r_array[i, j] = 9.205 * rc_lr**2 - 9.157 * rc_lr + 2.269
                        
                    elif mineral_type == 'zircon':
                        rho_r_array[i, j] = 1.25 * (r - 0.2)
                
                #calculate t_eq, prevent subzero indexing
                if j == 0:
                    break
                elif(r<1):
                    temp_mean = np.log(2 / (interp_tT[j - 1, 1] + interp_tT[j, 1]))
                    t_eq = np.exp(
                        C2 + (temp_mean - C3) * (((1 / r) - 1) ** alpha - C0) / C1
                    )

        return rho_r_array, interp_tT

    def guenthner_anneal(self):
        """
        Contains the zircon fission track annealing kinetic parameters reported in Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01). Values are passed to anneal method, which returns reduced track length 2D array that is converted to track density.

        Returns
        -------

        rho_r_array: 2D array of floats
            Reduced track densities for each time slice of an instance of the class' tT path.

        relevant_tT: 2D array of floats
            Condensed t-T path, time only goes as far back as is 'relevant' to damage annealing

        """
        C0 = 6.24534
        C1 = -0.11977
        C2 = -314.937
        C3 = -14.2868
        alpha = -0.05721
        total_anneal = 0.36/1.25+0.2
        rmr0 = None
        kappa = None

        kinetics_list = [C0, C1, C2, C3, alpha,total_anneal, rmr0, kappa]

        rho_r_array, relevant_tT = self.anneal(kinetics_list)

        return rho_r_array, relevant_tT
        
    def ketcham_anneal(self, rmr0=0.83):
        """
        Contains the apatite fission track annealing kinetic parameters reported in Ketcham et al., (2007) (https://doi.org/10.2138/am.2007.2281), which are used in the Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015) derivation of the apatite radiation damage and annealing model. Values are passed to anneal method, which returns reduced track length 2D array that is converted to track density.

        Parameters
        ----------
        
        rmr0: float
            Fitted parameter that converts the annealing of fission tracks in apatite B2 to any other apatite. Flowers et al. (2009) use a set value of 0.83 (default here), but natural apatites have a range that could be user specified.

        Returns
        -------

        rho_r_array: 2D array of floats
            Reduced track densities for each time slice of an instance of the class' tT path.
        
        relevant_tT: 2D array of floats
            Condensed t-T path, time only goes as far back as is 'relevant' to damage annealing

        """
        C0 = 0.39528
        C1 = 0.01073
        C2 = -65.12969
        C3 = -7.91715
        alpha = 0.04672

        kappa = 1.04 - rmr0
        B2_total_anneal = 0.55
        total_anneal = B2_total_anneal ** (1 / kappa) * (1 - rmr0) + rmr0

        kinetics_list = [C0, C1, C2, C3, alpha,total_anneal, rmr0, kappa]

        rho_r_array, relevant_tT = self.anneal(kinetics_list)
        
        return rho_r_array, relevant_tT
