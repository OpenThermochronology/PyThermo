""" 
tT_model.py

Particular approaches to forward modeling and plotting apatite and zircon (U-Th)/He date-effective uranium (eU) relationships.

"""
import matplotlib.pyplot as plt
from cycler import cycler
from .constants import np
from .crystal import apatite, zircon
from .tT_path import tT_path

class tT_model:
    def __init__(self, grain_in, tT_in, obs_data=None, temp_precision=5):
        """ 
        Constructor for tT_model class. Supports mineral types: 'apatite' and 'zircon'. Supports damage-diffusivity models: 'guenthner', kinetics for zircon system of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01), and 'flowers', kinetics for apatite system of Flowers et al. (2009) (https://doi.org/10.1016/j.gca.2009.01.015). Supports damage annealing models: 'guenthner_anneal', zircon fission track annealing parameters of Guenthner et al. (2013) (https://doi.org/10.2475/03.2013.01),'ketcham_anneal', apatite fission track annealing parameters of Ketcham et al., (2007) (https://doi.org/10.2138/am.2007.2281). Units: Date, date error, and time = Ma, temperature = degrees C, grain radii = micrometers, concentrations = ppm.

        Parameters
        ----------
        grain_in: data frame
            Grain input information. Each grain occupies one row. Columns consist of: mineral type (1st column), dates (2nd column), 2 sigma date error (3rd column), grain size (4th column), U concentration (5th column), Th concentration (6th column), Sm concentration (7th column), damage-diffusivity model (8th column), annealing model (9th column)
        
        tT_in: 2D array
            Time-temperature history, or step-heating inputs for forward models and arrhenius functions. Time is 1st column (young -> old for tT history) and temperature is 2nd column. Can contain multiple tT paths, ordered by column (spacing of 2).

        obs_data: optional 2D array
            Measured data for forward model comparison: measured grain dates (1st column), 2 sigma error (2nd column), eU concentrations (3rd column), and grain size (4th column). Default is 'None'.
        
        temp_precision: optional float
            Default minimal spacing of temperature steps in the interpolated tT path. Smaller numbers yield better precision in the diffusion solver (and date) at the cost of increased run-time. Default is 5.
        
        """
        self.__grain_in = grain_in
        self.__tT_in = tT_in
        self.__obs_data = obs_data
        self.__temp_precision = temp_precision
        self.__model_data = None
    
    def get_grain_in(self):
        return self.__grain_in
    
    def get_tT_in(self):
        return self.__tT_in
    
    def get_obs_data(self):
        return self.__obs_data
    
    def get_temp_precision(self):
        return self.__temp_precision
    
    def get_model_data(self):
        return self.__model_data
    
    def set_model_data(self,model_data):
        self.__model_data = model_data

    def forward(self, comp_type='size', model_num=1, std_grain=0, log2_nodes=8):
        """ 
        Runs a forward model for each individual tT path entry in class variable tT_in and the grain inputs in class variable grain_in.

        Parameters
        ----------
        comp_type: optional string
            Model comparison type for plot output. Can be either 'size' for grain size comparison, or 'model' for annealing or diffusion model comparison. Default is 'size'.

        model_num: optional int
            Number of model comparisons if comp_type = 'model'. Corresponds to number of curves per tT path. Default is 1.
                
        std_grain: optional float
            The user-defined 1 sigma standard deviation in grain size for grain size comparison runs. Can be used if obs_data is 'None'. Default is 0.
        
        log2_nodes: optional int
            The number of nodes (2**log2_nodes + 1) used in the Crank-Nicolson diffusion solver. Default is 8 (256 nodes + 1).

        Returns
        -------
        forward_fig: figure object
            Figure object containing tT paths and correspondind date-eU plots.

        """
        tT_in = self.__tT_in
        grains = self.__grain_in
        obs_data = self.__obs_data
        
        #don't assume that everyone will have the same column or row data frame names, use indexing throughout
        num_grains = grains.shape[0]
        num_paths = np.size(tT_in,1)//2
        
        #create 2D array for storing model output, with some error checking
        if comp_type == 'size':
            #determine mean, maximum, and minimum grain size 
            #if obs_data is present, use mean from measured, otherwise use grain size in grain_in
            if obs_data is not None and std_grain == 0:
                mean_size = np.mean(obs_data[:,3])
                max_size = mean_size + np.std(obs_data[:,3])
                min_size = mean_size - np.std(obs_data[:,3])                
            elif obs_data is not None:
                mean_size = np.mean(obs_data[:,3])
                max_size = mean_size + std_grain
                min_size = mean_size - std_grain
            else:
                mean_size = grains.iloc[:,3].mean()
                max_size = mean_size + std_grain
                min_size = mean_size - std_grain

            #now create the array
            if max_size > mean_size:
                model_data = np.zeros((num_grains*3,num_paths*3))
            else:
                model_data = np.zeros((num_grains,num_paths*3))
                
        elif comp_type == 'model':
            model_data = np.zeros((num_grains,num_paths*3))

        else:
            print("Incorrect comparison type entered")
            return None
        
        #calculate dates for each tT path
        for i in range(0,np.size(tT_in,1),2):
            #not all tT paths will be the same length, some will have zeros after oldest time to fill in 2D array
            #find the oldest time, use only that splice of the t and T columns
            non_zero = np.argwhere(tT_in[:,i])
            last_time = non_zero[-1][0]
            
            #interpolate tT path
            tT_slice = tT_in[:last_time+1,i:i+2]
            tT = tT_path(tT_slice,self.__temp_precision)
            tT.tT_interpolate()

            #get rho_r annealing and relevant tT vectors, same for all grains for a given tT path
            if 'guenthner' in grains.iloc[:,8].values: zirc_anneal,zirc_tT = tT.guenthner_anneal()
            if 'ketcham' in grains.iloc[:,8].values: ap_anneal,ap_tT = tT.ketcham_anneal()
            
            #calculate dates for each grain
            for j in range(num_grains):
                U_ppm = grains.iloc[j,4]
                Th_ppm = grains.iloc[j,5]
                Sm_ppm = grains.iloc[j,6]
                #Cooperdock et al. (2019) eU approximation (https://doi.org/10.5194/gchron-1-17-2019)
                eU_ppm = U_ppm + 0.238*Th_ppm + 0.0012*Sm_ppm

                if grains.iloc[j,0] == 'apatite':
                    if max_size > mean_size:
                        mean_grain = apatite(mean_size,log2_nodes,ap_tT,ap_anneal,U_ppm,Th_ppm,Sm_ppm)
                        std_plus_grain = apatite(max_size,log2_nodes,ap_tT,ap_anneal,U_ppm,Th_ppm,Sm_ppm)
                        std_minus_grain = apatite(min_size,log2_nodes,ap_tT,ap_anneal,U_ppm,Th_ppm,Sm_ppm)

                        #calculate dates
                        mean_date = mean_grain.flowers_date()
                        std_plus_date = std_plus_grain.flowers_date()
                        std_minus_date = std_minus_grain.flowers_date()

                        #add dates to array
                        model_data[j,i*3//2] = mean_date
                        model_data[j+num_grains,i*3//2] = std_plus_date
                        model_data[j+2*num_grains,i*3//2] = std_minus_date

                        #add eU concentrations to array
                        model_data[j,i*3//2+1] = eU_ppm
                        model_data[j+num_grains,i*3//2+1] = eU_ppm
                        model_data[j+2*num_grains,i*3//2+1] = eU_ppm

                        #add grain size to array
                        model_data[j,i*3//2+2] = mean_size
                        model_data[j+num_grains,i*3//2+2] = max_size
                        model_data[j+2*num_grains,i*3//2+2] = min_size

                    else:
                        mean_grain = apatite(mean_size,log2_nodes,ap_tT,ap_anneal,U_ppm,Th_ppm,Sm_ppm)

                        #calculate date
                        mean_date = mean_grain.flowers_date()

                        #add date to array
                        model_data[j,i*3//2] = mean_date

                        #add eU concentration to array
                        model_data[j,i*3//2+1] = eU_ppm

                        #add grain size to array
                        model_data[j,i*3//2+2] = mean_size

                elif grains.iloc[j,0] == 'zircon':
                    if max_size > mean_size:
                        mean_grain = zircon(mean_size,log2_nodes,zirc_tT,zirc_anneal,U_ppm,Th_ppm,Sm_ppm) 
                        std_plus_grain = zircon(max_size,log2_nodes,zirc_tT,zirc_anneal,U_ppm,Th_ppm,Sm_ppm)
                        std_minus_grain = zircon(min_size,log2_nodes,zirc_tT,zirc_anneal,U_ppm,Th_ppm,Sm_ppm)

                        #calculate dates
                        mean_date = mean_grain.guenthner_date()
                        std_plus_date = std_plus_grain.guenthner_date()
                        std_minus_date = std_minus_grain.guenthner_date()

                        #add dates to array
                        model_data[j,i*3//2] = mean_date
                        model_data[j+num_grains,i*3//2] = std_plus_date
                        model_data[j+2*num_grains,i*3//2] = std_minus_date

                        #add eU concentrations to array
                        model_data[j,i*3//2+1] = eU_ppm
                        model_data[j+num_grains,i*3//2+1] = eU_ppm
                        model_data[j+2*num_grains,i*3//2+1] = eU_ppm

                        #add grain size to array
                        model_data[j,i*3//2+2] = mean_size
                        model_data[j+num_grains,i*3//2+2] = max_size
                        model_data[j+2*num_grains,i*3//2+2] = min_size

                    else:
                        mean_grain = zircon(mean_size,log2_nodes,zirc_tT,zirc_anneal,U_ppm,Th_ppm,Sm_ppm)

                        #calculate date
                        mean_date = mean_grain.guenthner_date()

                        #add date to array
                        model_data[j,i*3//2] = mean_date

                        #add eU concentration to array
                        model_data[j,i*3//2+1] = eU_ppm

                        #add grain size to array
                        model_data[j,i*3//2+2] = mean_size

                else:
                    print('Improper mineral type used. Grain number ',j+1,' for tT path ',i+1,' was not modeled.')
                    continue

        self.set_model_data(model_data)
        forward_fig = self.date_eU_plot(model_data,num_grains,'size')

        return forward_fig

    def date_eU_plot(self, model_data, grain_num, comp_type):
        """ 
        Plot model date output on a date-eU plot, along with corresponding tT paths, and optional measured datasets.

        Parameters
        ----------

        model_data: 2D array of floats
            Array containing model (U-Th)/He dates (1st column), eU concentrations (2nd column), and grain sizes (3rd column). Can contain multipe date-eU curves. Curves ordered by row correspond to the same tT path (different grain sizes). Curves ordered by column (spacing of 3) correspond to different tT paths.
        
        grain_num: int
            Number of model grains, per mineral type, used in input. Specifically the number of different eU concentrations.
        
        comp_type: string
            The type of comparison. Either 'size' for grain size comparisons, or 'model' for different diffusion and/or annealing models.

        Returns
        -------

        dateeU_fig: figure object
            Figure object containing tT and date-eU plots to be saved. Keeps it basic, and allows for customization elsewhere.

        """
        tT_in = self.__tT_in
        obs_data = self.__obs_data

        #set up the figure with 2 subplots: time_temp  and date_eU
        dateeU_fig, (time_temp, date_eU) = plt.subplots(2, 1, figsize=[5,6], dpi=600)

        time_temp.set_xlabel('Time (Ma)')
        time_temp.set_ylabel('Temperature $\\degree$C')
        date_eU.set_xlabel('eU concentration (ppm)')
        date_eU.set_ylabel('(U-Th)/He Date (Ma)')

        #maximum values for setting plot axis dimensions
        date_max = 0
        eU_max = 0
        time_max = 0
        temp_max = 0

        #set up color strings, if greater than 8 date-eU/tT sets, switch to viridis gradational color scheme
        if np.size(model_data,1)/3 < 8:
            color_options = ['xkcd:black','xkcd:royal blue','xkcd:red','xkcd:sky','xkcd:lime','xkcd:dark purple','xkcd:rose','xkcd:grey']
        else:
            color_options = []

        #line option cycler for model comparisons
        line_options = cycler(line_style=['-','--',':','-.'])

        #plot date-eU trends
        #assumes that the date-eU trends for each tT plot are in 3-column groups, in this order: date, eU, grain size
        for i in range(0,np.size(model_data,1),3):
            
            if np.max(model_data[:,i]) > date_max: date_max = np.max(model_data[:,i])
            if np.max(model_data[:,i+1]) > eU_max: eU_max = np.max(model_data[:,i+1])

            if comp_type == 'size':
                #each grain size comparison has three stacked date-eU trends, in this order: mean, +1s, -1s
                dates_mean = model_data[0:grain_num,i]
                eUs_mean = model_data[0:grain_num,i+1]
                date_eU.plot(eUs_mean,dates_mean,linestyle='-',marker='',color=color_options[i//3])
                
                if np.size(model_data,0) > grain_num:
                    dates_2s_plus = model_data[grain_num:2*grain_num,i]
                    eUs_2s_plus = model_data[grain_num:2*grain_num,i+1]
                    dates_2s_minus = model_data[2*grain_num:3*grain_num,i]
                    eUs_2s_minus = model_data[2*grain_num:3*grain_num,i+1]

                    date_eU.plot(eUs_2s_minus,dates_2s_minus,linestyle='--',marker='',color=color_options[i//3])
                    date_eU.plot(eUs_2s_plus,dates_2s_plus,linestyle='--',marker='',color=color_options[i//3])

            elif comp_type == 'model':
                date_eU.set_prop_cycle(line_options)
                #each diffusion and/or annealing model output is stacked from low to high eU, allows for variable # of model comps
                for j in range(0,np.size(model_data,0),grain_num):
                    model_dates = model_data[j:j+grain_num,i]
                    model_eUs = model_data[j:j+grain_num,i+1]
                    date_eU.plot(model_eUs, model_dates,color=color_options[i//3])
            
        #plot tT paths
        #assumes that the tT paths are in 2-column groups, in this order: time (young -> old), temp
        for i in range(0,np.size(tT_in,1),2):

            #not all tT paths will be the same length, some will have zeros after oldest time to fill in 2D array
            #find the oldest time, use only that splice of the t and T columns
            non_zero = np.argwhere(tT_in[:,i])
            last_time = non_zero[-1][0]
            if np.max(tT_in[:last_time+1,i]) > time_max: time_max = np.max(tT_in[:last_time+1,i])
            if np.max(tT_in[:last_time+1,i+1]) > temp_max: temp_max = np.max(tT_in[:last_time+1,i+1])

            times = tT_in[:last_time+1,i]
            temps = tT_in[:last_time+1,i+1]

            time_temp.plot(times,temps,color=color_options[i//2])

        #add in measured data, if any
        if obs_data is not None:
            date_eU.errorbar(obs_data[:,2],obs_data[:,0],yerr=obs_data[:,1],linestyle = '',marker = 'o',ms = 5, mfc = 'k',mec = 'k')
            
            if np.max(obs_data[:,0]) > date_max: date_max = np.max(obs_data[:,0])
            if np.max(obs_data[:,2]) > eU_max: eU_max = np.max(obs_data[:,2])
        
        #set up axis dimensions
        
        #scale it to inputs, round up to the nearest 10, 50, or 250 and add a bit more
        if date_max < 100:
            date_max = int(np.ceil(date_max/10)*10) + 10
        elif date_max < 500:
            date_max = int(np.ceil(date_max/50)*50) + 50
        else:
            date_max = int(np.ceil(date_max/250)*250) + 250
        
        if eU_max < 100:
            eU_max = int(np.ceil(eU_max/10)*10) + 10
        elif eU_max < 500:
            eU_max = int(np.ceil(eU_max/50)*50) + 50
        else:
            eU_max = int(np.ceil(eU_max/250)*250) + 250

        if time_max < 100:
            time_max = int(np.ceil(time_max/10)*10) + 10
        elif time_max < 500:
            time_max = int(np.ceil(time_max/50)*50) + 50
        else:
            time_max = int(np.ceil(time_max/250)*250) + 250

        #round up to the nearest 50 for temperature
        temp_max = int(np.ceil(temp_max/50)*50)

        time_temp.set_xticks(np.arange(0,time_max,time_max/10))
        time_temp.set_yticks(np.arange(0,temp_max,temp_max/10))
        time_temp.set_xlim(time_max,0)
        time_temp.set_ylim(temp_max,0)
        date_eU.set_xticks(np.arange(0,eU_max,eU_max/10))
        date_eU.set_yticks(np.arange(0,date_max,date_max/10))
        
        dateeU_fig.tight_layout()
        return dateeU_fig
