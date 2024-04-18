import pythermo as pyt
import pytest
import numpy as np

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

def test_alpha_ejection():
    pass

def test_CN_diffusion():
    pass

def test_He_date():
    pass

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
    damage = ap.guenthner_damage()
    relevant_tT = ap.get_relevant_tT()
    assert np.size(damage) == np.size(relevant_tT,0)
    assert np.isnan(damage).any() == False
    assert np.any(damage < 0) == False

def test_flowers_date(ap):
    ap_date = ap.flowers_date()
    assert ap_date > 0
    assert ap_date < 500