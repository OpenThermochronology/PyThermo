import pythermo as pyt
import pytest
import numpy as np
import math

@pytest.fixture(scope='session', autouse=True)
def warmup_numba():
    #Force Numba JIT compilation once before any tests run, using a small grain.
    tT_in = np.array([[0, 20], [100, 50], [200, 20]])
    tT = pyt.tT_path(tT_in)
    
    #warmup guenthner path
    zirc_anneal, zirc_tT = tT.guenthner_anneal()
    zirc = pyt.zircon(30, 6, zirc_tT, zirc_anneal, 100, 50)
    zirc.zirc_date('guenthner', 'guenthner')
    
    #warmup flowers path
    ap_anneal, ap_tT = tT.ketcham_anneal()
    ap = pyt.apatite(30, 6, ap_tT, ap_anneal, 100, 50, Sm_ppm=10)
    ap.ap_date('flowers', 'flowers')

@pytest.fixture()
def ap():
    tT_in = np.array([[0, 20], [250, 100], [500, 20]])
    tT = pyt.tT_path(tT_in)
    ap_anneal,ap_tT = tT.ketcham_anneal()
    return pyt.apatite(
        60,
        8,
        ap_tT,
        ap_anneal,
        100,
        50,
        Sm_ppm=10,
    )

@pytest.fixture()
def zirc():
    tT_in = np.array([[0, 20], [250, 200], [500, 20]])
    tT = pyt.tT_path(tT_in)
    zirc_anneal,zirc_tT = tT.guenthner_anneal()
    return pyt.zircon(
        60,
        8,
        zirc_tT,
        zirc_anneal,
        1000,
        500,
    )

@pytest.fixture()
def zirc_no_sm():
    tT_in = np.array([[0, 20], [250, 200], [500, 20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    zirc_anneal, zirc_tT = tT.guenthner_anneal()
    return pyt.zircon(
        60, 
        8, 
        zirc_tT, 
        zirc_anneal, 
        1000, 
        500, 
        Sm_ppm=0
    )

#alpha ejection tests

def test_apatite_alpha_ejection(ap):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    expected_nodes = ap.get_nodes()

    assert (
        len(aej_U238) == len(aej_U235) 
        and len(aej_U238) == len(aej_Th) 
        and len(aej_U238) == len(aej_Sm) 
        and len(aej_U235) == len(aej_Th) 
        and len(aej_U235) == len(aej_Sm) 
        and len(aej_Th) == len(aej_Sm) 
        and len(aej_U238) == expected_nodes
    )
    assert all(i >= 0 for i in aej_U238)
    assert all(i >= 0 for i in aej_U235)
    assert all(i >= 0 for i in aej_Th)
    assert all(i >= 0 for i in aej_Sm)

    assert ft_U238 >= 0 
    assert ft_U235 >= 0 
    assert ft_Th >= 0 
    assert ft_Sm >= 0 
    
def test_zircon_alpha_ejection(zirc):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    expected_nodes = zirc.get_nodes()

    assert (
        len(aej_U238) == len(aej_U235) 
        and len(aej_U238) == len(aej_Th) 
        and len(aej_U238) == len(aej_Sm) 
        and len(aej_U235) == len(aej_Th) 
        and len(aej_U235) == len(aej_Sm) 
        and len(aej_Th) == len(aej_Sm) 
        and len(aej_U238) == expected_nodes
    )
    assert all(i >= 0 for i in aej_U238)
    assert all(i >= 0 for i in aej_U235)
    assert all(i >= 0 for i in aej_Th)
    assert all(i >= 0 for i in aej_Sm)

    assert ft_U238 >= 0 
    assert ft_U235 >= 0 
    assert ft_Th >= 0 
    assert ft_Sm >= 0

def test_zero_sm_alpha_ejection(zirc_no_sm):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc_no_sm.zircon_alpha_ejection()
    
    assert np.all(aej_Sm == 0)
    assert corr_factors['total_Sm'] == 0.0
    assert corr_factors['ft_Sm'] == 0.0 

#diffusion solver tests

def test_ap_CN_diffusion(ap):
    
    damage = ap.flowers_damage()
    assert len(damage)+1 == np.size(ap.get_relevant_tT(), 0)
    
    diff_list = ap.flowers_diffs(damage)
    assert len(diff_list) == np.size(ap.get_relevant_tT(), 0) - 1

    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    r_step = 60 / (257.5)
    He_profile = ap.CN_diffusion(
        257,
        r_step,
        ap.get_relevant_tT(),
        diff_list,
        aej_U238,
        aej_U235,
        aej_Th,
        aej_Sm,
    )
    He_profile = np.array(He_profile)

    assert np.isnan(He_profile).any() == False
    assert np.any(He_profile < 0) == False

def test_zirc_CN_diffusion(zirc):

    damage = zirc.guenthner_damage()
    assert len(damage) + 1 == np.size(zirc.get_relevant_tT(), 0)
    
    diff_list = zirc.guenthner_diffs(damage)
    assert len(diff_list) == np.size(zirc.get_relevant_tT(), 0) - 1

    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    r_step = 60 / (257.5)
    He_profile = zirc.CN_diffusion(
        257,
        r_step,
        zirc.get_relevant_tT(),
        diff_list,
        aej_U238,
        aej_U235,
        aej_Th,
        aej_Sm,
    )
    He_profile = np.array(He_profile)

    assert np.isnan(He_profile).any() == False
    assert np.any(He_profile < 0) == False

def test_guenthner_diffs(zirc):
    damage = zirc.guenthner_damage()
    diff_list = zirc.guenthner_diffs(damage)
    diff_list = np.array(diff_list)
    
    assert len(diff_list) == np.size(zirc.get_relevant_tT(), 0) - 1
    assert np.isnan(diff_list).any() == False
    assert np.any(diff_list <= 0) == False  # diffusivities must be positive

def test_flowers_diffs(ap):
    damage = ap.flowers_damage()
    diff_list = ap.flowers_diffs(damage)
    diff_list = np.array(diff_list)

    assert len(diff_list) == np.size(ap.get_relevant_tT(), 0) - 1
    assert np.isnan(diff_list).any() == False
    assert np.any(diff_list <= 0) == False

def test_mp_diffs(zirc):
    damage = zirc.guenthner_damage()
    fast_diffs, lat_diffs, bulk_diffs = zirc.mp_diffs(damage)
    
    assert len(fast_diffs) == np.size(zirc.get_relevant_tT(), 0) - 1
    assert len(lat_diffs) == np.size(zirc.get_relevant_tT(), 0) - 1
    assert len(bulk_diffs) == np.size(zirc.get_relevant_tT(), 0) - 1
    assert np.any(fast_diffs <= 0) == False
    assert np.any(lat_diffs <= 0) == False
    assert np.any(bulk_diffs <= 0) == False
    
    #fast path should be faster than lattice
    assert np.all(fast_diffs >= lat_diffs)

#profile wrapper tests

def test_zirc_CN_profile(zirc):
    damage = zirc.guenthner_damage()
    diffs = zirc.guenthner_diffs(damage)
    init_He = zirc._aej_U238  #use a dummy alpha ejection profile as initial condition
    
    bulk_He, total_He = zirc.CN_profile(diffs, init_He)
    bulk_He = np.array(bulk_He)

    assert np.isnan(bulk_He).any() == False
    assert np.any(bulk_He < 0) == False
    assert total_He >= 0

def test_ap_CN_profile(ap):
    damage  = ap.flowers_damage()
    diffs   = ap.flowers_diffs(damage)
    init_He = ap._aej_U238  #use a dummy alpha ejection profile as initial condition

    bulk_He, total_He = ap.CN_profile(diffs, init_He)
    bulk_He = np.array(bulk_He)

    assert np.isnan(bulk_He).any() == False
    assert np.any(bulk_He < 0) == False
    assert total_He >= 0

#He date method tests

def test_ap_He_date(ap):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = ap.apatite_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    date_guess = 55500000
    lambda_232_yr = np.log(2) / (1.405 * 10**10) 
    lambda_235_yr = np.log(2) / (7.0381 * 10**8) 
    lambda_238_yr = np.log(2) / (4.4683 * 10**9)
    lambda_147_yr = 6.54 * 10**-12

    total_He = (
        ft_U238 
        * (np.exp(lambda_238_yr * date_guess) - 1)
        + ft_U235
        * (np.exp(lambda_235_yr * date_guess) - 1)
        + ft_Th
        * (np.exp(lambda_232_yr * date_guess) - 1)
        + ft_Sm
        * (np.exp(lambda_147_yr * date_guess) -1)
    )
    date = ap.He_date(total_He, corr_factors)

    assert math.isclose(date * 10**6, date_guess, rel_tol=.0001)

def test_zirc_He_date(zirc):
    aej_U238, aej_U235, aej_Th, aej_Sm, corr_factors = zirc.zircon_alpha_ejection()
    ft_U238 = corr_factors['total_U238'] * corr_factors['ft_U238']
    ft_U235 = corr_factors['total_U235'] * corr_factors['ft_U235']
    ft_Th = corr_factors['total_Th'] * corr_factors['ft_Th']
    ft_Sm = corr_factors['total_Sm'] * corr_factors['ft_Sm']

    date_guess = 55500000
    lambda_232_yr = np.log(2) / (1.405 * 10**10) 
    lambda_235_yr = np.log(2) / (7.0381 * 10**8) 
    lambda_238_yr = np.log(2) / (4.4683 * 10**9)
    lambda_147_yr = 6.54 * 10**-12

    total_He = (
        ft_U238
        * (np.exp(lambda_238_yr * date_guess) - 1)
        + ft_U235
        * (np.exp(lambda_235_yr * date_guess) - 1)
        + ft_Th
        * (np.exp(lambda_232_yr * date_guess) - 1)
        + ft_Sm
        * (np.exp(lambda_147_yr * date_guess) - 1)
    )
    date = zirc.He_date(total_He, corr_factors)

    assert math.isclose(date * 10**6, date_guess, rel_tol=.0001)

def test_guenthner_damage(zirc):
    damage = zirc.guenthner_damage()
    relevant_tT = zirc.get_relevant_tT()
    assert np.size(damage) + 1 == np.size(relevant_tT, 0)
    assert np.isnan(damage).any() == False
    assert np.any(damage < 0) == False

def test_zirc_date(zirc):
    zirc_date = zirc.zirc_date('guenthner','guenthner')[0]
    assert zirc_date > 0
    assert zirc_date < 500

def test_flowers_damage(ap):
    damage = ap.flowers_damage()
    damage = np.array(damage)
    relevant_tT = ap.get_relevant_tT()
    assert np.size(damage) + 1 == np.size(relevant_tT, 0)
    assert np.isnan(damage).any() == False
    assert np.any(damage < 0) == False

def test_ap_date(ap):
    ap_date = ap.ap_date('flowers','flowers')[0]
    assert ap_date > 0
    assert ap_date < 500

#integration test

def test_integrate_profile_negative(zirc):
    #passing a profile with large negatives should return zeros
    bad_profile = np.full(zirc.get_nodes(), -1e10)
    profile, total = zirc._integrate_profile(bad_profile)
    
    assert total == 0.0
    assert np.all(profile == 0)

#getter method tests

def test_zirc_getters(zirc):
    assert zirc.get_radius() == 60
    assert zirc.get_log2_nodes() == 8
    assert zirc.get_nodes() == 2**8 + 1
    assert zirc.get_U_ppm() == 1000
    assert zirc.get_Th_ppm() == 500
    assert zirc.get_Sm_ppm() == 0
    assert zirc.get_r_step() == pytest.approx(60 / (257.5))

def test_ap_getters(ap):
    assert ap.get_radius() == 60
    assert ap.get_log2_nodes() == 8
    assert ap.get_nodes() == 2**8 + 1
    assert ap.get_U_ppm() == 100
    assert ap.get_Th_ppm() == 50
    assert ap.get_Sm_ppm() == 10
    assert ap.get_r_step() == pytest.approx(60 / (257.5))