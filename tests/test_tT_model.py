import pythermo as pyt
import pandas as pd
import numpy as np

def test_ap_forward():
    tT_in = np.array([[0,20],[250,150],[500,20]])

    dict = {
        'mineral type' : ['apatite', 'apatite', 'apatite'], 
        'age' : [60, 80, 90],
        'error' : [2, 2, 2],
        'size' : [60, 60, 60],
        'U ppm' : [10, 50, 150],
        'Th ppm' : [10, 10, 10],
        'Sm ppm' : [10, 10, 10],
        'diffusion model' : ['flowers', 'flowers', 'flowers'],
        'annealing model' : ['ketcham', 'ketcham', 'ketcham'],
    }

    grain_in = pd.DataFrame(
        data=dict,
        columns = ['mineral type', 'age', 'error','size','U ppm', 'Th ppm', 'Sm ppm', 'diffusion model', 'annealing model'],
        index = ['sample 1', 'sample 2', 'sample 3'],
    )
    
    obs_data = np.array(
        [[62.8, 1.2, 60, 80], [70.1, 2.1, 20, 85], [55.2, 0.9, 120, 65]]
    )

    model_test_no_obs = pyt.tT_model(grain_in, tT_in)
    model_test_no_obs_std = pyt.tT_model(grain_in, tT_in)
    model_test_obs = pyt.tT_model(grain_in, tT_in, obs_data)

    model_test_no_obs.forward()
    model_data = model_test_no_obs.get_model_data()
    assert np.size(model_data, 0) == 3
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert np.all(model_data[:, 2] == 60)

    model_test_no_obs_std.forward(std_grain=10)
    model_data = model_test_no_obs_std.get_model_data()
    assert np.size(model_data, 0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert (
        model_data[0, 2] == 60 
        and model_data[3, 2] == 70 
        and model_data[6, 2] == 50
    )

    model_test_obs.forward()
    model_data = model_test_obs.get_model_data()
    assert np.size(model_data,0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False

    model_test_obs.forward(std_grain=10)
    model_data = model_test_obs.get_model_data()
    assert np.size(model_data,0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert (
        model_data[0, 2] == np.mean(obs_data[:,3]) 
        and model_data[3,2] == np.mean(obs_data[:, 3]) + 10 
        and model_data[6, 2] == np.mean(obs_data[:, 3]) - 10
    )

def test_zirc_forward():
    tT_in = np.array([[0, 20], [250, 150], [500, 20]])
    
    dict = {
        'mineral type' : ['zircon', 'zircon', 'zircon'], 
        'age' : [60, 80, 90],
        'error' : [2, 2, 2],
        'size' : [60, 60, 60],
        'U ppm' : [100, 500, 1000],
        'Th ppm' : [10, 10, 10],
        'Sm ppm' : [0, 0, 0],
        'diffusion model' : ['guenthner', 'guenthner', 'guenthner'],
        'annealing model' : ['guenthner', 'guenthner', 'guenthner'],
    }

    grain_in = pd.DataFrame(
        data=dict,
        columns = ['mineral type', 'age', 'error','size','U ppm', 'Th ppm', 'Sm ppm', 'diffusion model', 'annealing model'],
        index = ['sample 1', 'sample 2', 'sample 3'],
    )
    
    obs_data = np.array(
        [[62.8, 1.2, 250, 80], [70.1, 2.1, 1500, 85], [55.2 , 0.9, 500, 65]]
    )
    
    model_test_no_obs = pyt.tT_model(grain_in,tT_in)
    model_test_no_obs_std = pyt.tT_model(grain_in,tT_in)
    model_test_obs = pyt.tT_model(grain_in,tT_in,obs_data)

    model_test_no_obs.forward()
    model_data = model_test_no_obs.get_model_data()
    assert np.size(model_data, 0) == 3
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert np.all(model_data[:, 2] == 60)

    model_test_no_obs_std.forward(std_grain=10)
    model_data = model_test_no_obs_std.get_model_data()
    assert np.size(model_data,0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert (
        model_data[0, 2] == 60 
        and model_data[3, 2] == 70 
        and model_data[6, 2] == 50
    )

    model_test_obs.forward()
    model_data = model_test_obs.get_model_data()
    assert np.size(model_data,0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False

    model_test_obs.forward(std_grain=10)
    model_data = model_test_obs.get_model_data()
    assert np.size(model_data,0) == 9
    assert np.isnan(model_data).any() == False
    assert np.any(model_data < 0) == False
    assert np.any(model_data[:, 1] < 0) == False
    assert np.any(model_data[:, 2] < 0) == False
    assert (
        model_data[0, 2] == np.mean(obs_data[:, 3]) 
        and model_data[3, 2] == np.mean(obs_data[:, 3]) + 10 
        and model_data[6, 2] == np.mean(obs_data[:, 3]) - 10
    )
