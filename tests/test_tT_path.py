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