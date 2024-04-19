import pythermo as pyt
import pytest
import numpy as np
import math

@pytest.fixture()
def ap():
    tT_in = np.array([[0,20],[250,100],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    ap_anneal,ap_tT = tT.ketcham_anneal()
    return pyt.apatite(60,8,ap_tT,ap_anneal,100,50,Sm_ppm=10)

@pytest.fixture()
def zirc():
    tT_in = np.array([[0,20],[250,200],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    zirc_anneal,zirc_tT = tT.ketcham_anneal()
    return pyt.zircon(60,8,zirc_tT,zirc_anneal,1000,500)

def test_apatite_alpha_ejection(ap):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    assert len(aej_U238) == len(aej_U235) and len(aej_U238) == len(aej_Th) and len(aej_U238) == len(aej_Sm) and len(aej_U235) == len(aej_Th) and len(aej_U235) == len(aej_Sm) and len(aej_Th) == len(aej_Sm)
    assert all(i >= 0 and i<=1 for i in aej_U238)
    assert all(i >= 0 and i<=1 for i in aej_U235)
    assert all(i >= 0 and i<=1 for i in aej_Th)
    assert all(i >= 0 and i<=1 for i in aej_Sm)

    assert ft_U238 > 0 and ft_U238 < 1
    assert ft_U235 > 0 and ft_U235 < 1
    assert ft_Th > 0 and ft_Th < 1
    assert ft_Sm > 0 and ft_Sm < 1
    

def test_zircon_alpha_ejection(zirc):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    assert len(aej_U238) == len(aej_U235) and len(aej_U238) == len(aej_Th) and len(aej_U238) == len(aej_Sm) and len(aej_U235) == len(aej_Th) and len(aej_U235) == len(aej_Sm) and len(aej_Th) == len(aej_Sm)
    assert all(i >= 0 and i<=1 for i in aej_U238)
    assert all(i >= 0 and i<=1 for i in aej_U235)
    assert all(i >= 0 and i<=1 for i in aej_Th)
    assert all(i >= 0 and i<=1 for i in aej_Sm)

    assert ft_U238 > 0 and ft_U238 < 1
    assert ft_U235 > 0 and ft_U235 < 1
    assert ft_Th > 0 and ft_Th < 1
    assert ft_Sm > 0 and ft_Sm < 1

def test_ap_CN_diffusion(ap):
    tT_in = np.array([[0,20],[250,100],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    ap_anneal,ap_tT = tT.ketcham_anneal()
    damage = ap.flowers_damage()
    gas_constant = 0.008314462618
    
    #Flowers et al. 2009 damage-diffusivity equation parameters
    omega = 10**-22
    psi = 10**-13
    E_trap = 34.0 #kJ/mol
    E_L = 122.3 #kJ/mol
    D0_L_a2 = np.exp(9.733) #1/s

    #convert D0_L_a2 to micrometers2/s using crystal radius
    D0_L = D0_L_a2 * 60**2

    diff_list = [(D0_L * np.exp(-E_L/(gas_constant*0.5*(ap_tT[i,1]+ap_tT[i+1,1]))))/ (((psi*damage[i] + omega*damage[i]**3) * np.exp(E_trap/(gas_constant*0.5*(ap_tT[i,1]+ap_tT[i+1,1])))) + 1) for i in range(len(damage))]

    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    r_step = 60/(257.5)
    He_profile = ap.CN_diffusion(257,r_step,ap_tT,diff_list,aej_U238,aej_U235,aej_Th,aej_Sm)

    assert np.isnan(He_profile).any() == False
    assert np.any(He_profile < 0) == False

def test_zirc_CN_diffusion(zirc):
    tT_in = np.array([[0,20],[250,100],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    zirc_anneal, zirc_tT = tT.guenthner_anneal()
    damage = zirc.guenthner_damage()
    gas_constant = 0.008314462618
    radius = 60

    #Guenthner et al. (2013) diffusion equation parameters, Eas are in kJ/mol, D0s converted to microns2/s
    Ea_l = 165.0
    D0_l = 193188.0 * 10**8 
    D0_N17 = 0.0034 * 10**8
    Ea_N17 = 71.0 

    #g amorphized per alpha event
    Ba = 5.48E-19
    interconnect = 3.0

    #empirical constraints for damage chains from Ketcham et al. (2013) (https://doi.org/10.2138/am.2013.4249) 
    #track surface to volume ratio in nm^-1 
    SV=1.669 
    #mean unidirectional length of travel until damage zone in a zircon with 1e14 alphas/g, in nm
    lint_0=45920.0

    #calculate diffusivities at each time step, modified equation 8 in Guenthner et al. (2013, units are in micrometers2/s; minimal diffusivity allowed equivalent to zircons with 1e14 alphas/g), prevents divide by zero in diffusivity calculation
    diff_list = [radius**2 * (((radius**2*np.exp(-Ba*damage[i]*interconnect)**3*(lint_0/(4.2/((1-np.exp(-Ba*damage[i]))*SV)-2.5))**2) /(D0_l * np.exp(-Ea_l/(gas_constant*((zirc_tT[i,1]+zirc_tT[i+1,1])/2))))) + ((radius**2*(1-np.exp(-Ba*damage[i]*interconnect)))**3 /(D0_N17*np.exp(-Ea_N17/(gas_constant*((zirc_tT[i,1]+zirc_tT[i+1,1])/2))))))**-1 if damage[i] >= 10**14 else radius**2 * (((radius**2)/(D0_l * np.exp(-Ea_l/(gas_constant*((zirc_tT[i,1]+zirc_tT[i+1,1])/2))))))**-1  for i in range(len(damage))]

    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    r_step = 60/(257.5)
    He_profile = zirc.CN_diffusion(257,r_step,zirc_tT,diff_list,aej_U238,aej_U235,aej_Th,aej_Sm)

    assert np.isnan(He_profile).any() == False
    assert np.any(He_profile < 0) == False

def test_ap_He_date(ap):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    date_guess = 55500000
    lambda_232_yr = np.log(2)/(1.405*10**10) 
    lambda_235_yr = np.log(2)/(7.0381*10**8) 
    lambda_238_yr = np.log(2)/(4.4683*10**9)
    lambda_147_yr = 6.54*10**-12

    total_He = ft_U238*(np.exp(lambda_238_yr*date_guess)-1)+ft_U235*(np.exp(lambda_235_yr*date_guess)-1)+ft_Th*(np.exp(lambda_232_yr*date_guess)-1)+ft_Sm*(np.exp(lambda_147_yr*date_guess)-1)
    date = zirc.He_date(total_He)

    assert math.isclose(date,date_guess,rel_tol=.001)


def test_zirc_He_date(zirc):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    date_guess = 55500000
    lambda_232_yr = np.log(2)/(1.405*10**10) 
    lambda_235_yr = np.log(2)/(7.0381*10**8) 
    lambda_238_yr = np.log(2)/(4.4683*10**9)
    lambda_147_yr = 6.54*10**-12

    total_He = ft_U238*(np.exp(lambda_238_yr*date_guess)-1)+ft_U235*(np.exp(lambda_235_yr*date_guess)-1)+ft_Th*(np.exp(lambda_232_yr*date_guess)-1)+ft_Sm*(np.exp(lambda_147_yr*date_guess)-1)
    date = zirc.He_date(total_He)

    assert math.isclose(date,date_guess,rel_tol=.001)

def test_guenthner_damage(zirc):
    damage = zirc.guenthner_damage()
    relevant_tT = zirc.get_relevant_tT()
    assert np.size(damage) == np.size(relevant_tT,0)
    assert np.isnan(damage).any() == False
    assert np.any(damage < 0) == False

def test_guenthner_date(zirc):
    zirc_date = zirc.guenthner_date()
    assert zirc_date > 0
    assert zirc_date < 500

def test_flowers_damage(ap):
    damage = ap.flowers_damage()
    relevant_tT = ap.get_relevant_tT()
    assert np.size(damage) == np.size(relevant_tT,0)
    assert np.isnan(damage).any() == False
    assert np.any(damage < 0) == False

def test_flowers_date(ap):
    ap_date = ap.flowers_date()
    assert ap_date > 0
    assert ap_date < 500