import pythermo as pyt
import pytest
import numpy as np

@pytest.fixture()
def tT():
    tT_in = np.array([[0, 20], [250, 150], [500, 20]])
    return pyt.tT_path(tT_in)

def test_tT_interpolate(tT):
    interp_tT = tT.tT_interpolate()
    
    assert np.all(interp_tT[: -1, 0] >= interp_tT[1:, 0])
    assert np.isnan(interp_tT).any() == False
    assert np.any(interp_tT < 0) == False

def test_tT_interpolate_expands(tT):
    interp_tT = tT.tT_interpolate()
    assert np.size(interp_tT, 0) > np.size(tT.get_tTin(), 0)

def test_tT_interpolate_endpoints(tT):
    interp_tT = tT.tT_interpolate()
    
    #time should start at 0
    assert interp_tT[-1, 0] == 0.0
    
    #temperature endpoints should match input (converted to Kelvin)
    assert interp_tT[-1, 1] == pytest.approx(20 + 273.15)

def test_tT_interpolate_precision():
    tT_in = np.array([[0, 20], [250, 150], [500, 20]])
    
    tT_coarse = pyt.tT_path(tT_in, temp_precision=20, rate_acceleration=2.0)
    tT_fine   = pyt.tT_path(tT_in, temp_precision=1,  rate_acceleration=1.1)
    
    interp_coarse = tT_coarse.tT_interpolate()
    interp_fine   = tT_fine.tT_interpolate()
    
    #finer precision should produce more time steps
    assert np.size(interp_fine, 0) > np.size(interp_coarse, 0)

def test_guenthner_anneal(tT):
    rho_r_array, relevant_tT = tT.guenthner_anneal()

    assert (
        np.size(rho_r_array, 0) == np.size(relevant_tT, 0) 
        and np.size(rho_r_array, 1) == np.size(relevant_tT, 0)
    )
    assert np.isnan(rho_r_array).any() == False
    assert np.any(rho_r_array < 0) == False
    assert np.any(rho_r_array > 1) == False

def test_ketcham_anneal(tT):
    rho_r_array, relevant_tT = tT.ketcham_anneal()

    assert (
        np.size(rho_r_array, 0) == np.size(relevant_tT, 0) 
        and np.size(rho_r_array, 1) == np.size(relevant_tT, 0)
    )
    assert np.isnan(rho_r_array).any() == False
    assert np.any(rho_r_array < 0) == False
    assert np.any(rho_r_array > 1) == False

def test_anneal_bad_kinetics(tT):
    with pytest.raises(ValueError):
        tT.anneal([1.0, 2.0, 3.0])  #too few kinetics values

#getter tests

def test_tT_getters(tT):
    tT_in = np.array([[0, 20], [250, 150], [500, 20]])
    assert np.array_equal(tT.get_tTin(), tT_in)
    assert tT.get_precision()    == 5
    assert tT.get_acceleration() == 1.5