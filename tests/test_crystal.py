import pythermo as pyt
import pytest
import numpy as np

@pytest.fixture()
def ap():
    tT_in = np.array([[0,20],[250,100],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    ap_anneal,ap_tT = tT.ketcham_anneal()
    return pyt.apatite(60,8,ap_tT,100,50,Sm_ppm=10)

@pytest.fixture()
def zirc():
    tT_in = np.array([[0,20],[250,100],[500,20]])
    tT = pyt.tT_path(tT_in)
    tT.tT_interpolate()
    zirc_anneal,zirc_tT = tT.ketcham_anneal()
    return pyt.zircon(60,8,zirc_tT,1000,500)

def test_alpha_ejection():
    pass

def test_CN_diffusion():
    pass

def test_He_date():
    pass

def test_guenthner_damage(zirc):
    pass

def test_guenthner_date(zirc):
    pass

def test_flowers_damage(ap):
    pass

def test_flowers_date(ap):
    pass