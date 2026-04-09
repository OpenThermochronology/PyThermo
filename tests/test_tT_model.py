import pythermo as pyt
import pandas as pd
import numpy as np

#useful helper functions

def make_grains(mineral="zircon", n=1, dam_model="guenthner",
                diff_model="guenthner", size=60.0,
                U_ppm=100.0, Th_ppm=50.0, Sm_ppm=10.0, age_ma=100.0):
    rows = [{
        0: mineral,          # mineral type
        1: age_ma,           # dose age (Ma)
        2: None,             # unused
        3: size,             # radius (µm)
        4: U_ppm,
        5: Th_ppm,
        6: Sm_ppm,
        7: diff_model,
        8: dam_model,
    }] * n
    return pd.DataFrame(rows)


def make_tT(steps=5, temp_c=300.0, duration_s=3600.0):
    return np.array([[duration_s, temp_c]] * steps)


def make_diff_params(init_style="distribute"):
    return {
        "kappa_1": 1e-3,
        "kappa_2": 1e-5,
        "f":       0.1,          # overwritten internally
        "init_style": init_style,
    }


def make_model(mineral="zircon", n_grains=1, dam_model="guenthner",
               diff_model="guenthner", n_steps=5, temp_c=300.0,
               age_ma=100.0):
    
    grains = make_grains(mineral=mineral, n=n_grains,
                         dam_model=dam_model, diff_model=diff_model,
                         age_ma=age_ma)
    tT = make_tT(steps=n_steps, temp_c=temp_c)
    return pyt.tT_model(grains_df=grains, tT=tT)


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
        'annealing model' : ['flowers', 'flowers', 'flowers'],
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

#step_degas tests

# ── 1. Output shape ───────────────────────────────────────────────────────────

class TestOutputShape:
    def test_returns_two_lists(self):
        model = make_model()
        result = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert isinstance(result, tuple) and len(result) == 2

    def test_list_length_equals_num_grains(self):
        N = 3
        model = make_model(n_grains=N)
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == N and len(pr) == N

    def test_frac_loss_array_has_three_columns(self):
        model = make_model(n_steps=4)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl[0].shape == (4, 3)

    def test_profile_array_columns(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "MP", 1e-6)
        assert pr[0].shape[1] == 3

    def test_profile_rows_match_nodes(self):
        nodes = 2**8 + 1
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "MP", 1e-6, log2_nodes=8)
        assert pr[0].shape[0] == nodes

    def test_no_nan_in_outputs(self):
        model = make_model()
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert not np.any(np.isnan(fl[0]))
        assert not np.any(np.isnan(pr[0]))

    def test_no_inf_in_outputs(self):
        model = make_model()
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert not np.any(np.isinf(fl[0]))


# ── 2. Input validation ───────────────────────────────────────────────────────

class TestInputValidation:
    def test_invalid_mineral_returns_none(self):
        model = make_model(mineral="quartz")
        result = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert result is None

    def test_apatite_runs_without_error(self):
        model = make_model(mineral="apatite", dam_model="flowers",
                           diff_model="flowers")
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 1

    def test_zircon_runs_without_error(self):
        model = make_model(mineral="zircon")
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 1

    def test_custom_init_profile_used(self):
        """Supplying init_profile should bypass the internal He calculator."""
        model = make_model()
        nodes = 2**8 + 1
        custom = np.ones(nodes) * 1e10
        fl_default, _ = model.step_degas(make_diff_params(), "CN", 1e-6,
                                         log2_nodes=8)
        fl_custom, _ = model.step_degas(make_diff_params(), "CN", 1e-6,
                                        init_profile=custom, log2_nodes=8)
        # Different initial He → different fractional loss trajectory
        assert not np.allclose(fl_default[0][:, 1], fl_custom[0][:, 1])

    def test_custom_grain_diffs_used(self):
        model = make_model(n_steps=3)
        # Artificially large diffusivities should increase fractional loss
        grain_diffs = np.full((3, 3), 1e-10)
        fl_default, _ = model.step_degas(make_diff_params(), "MP", 1e-6)
        fl_fast, _ = model.step_degas(make_diff_params(), "MP", 1e-6,
                                      grain_diffs=grain_diffs)
        assert fl_fast[0][-1, 1] > fl_default[0][-1, 1]


# ── 3. Fractional loss physics ────────────────────────────────────────────────

class TestFractionalLoss:
    def test_frac_loss_between_zero_and_one(self):
        model = make_model(n_steps=10, temp_c=400.0)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        vals = fl[0][:, 1]
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_frac_loss_monotonically_nondecreasing(self):
        model = make_model(n_steps=10, temp_c=350.0)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        vals = fl[0][:, 1]
        assert np.all(np.diff(vals) >= -1e-9)   # allow tiny float drift

    def test_low_temp_gives_near_zero_loss(self):
        model = make_model(n_steps=3, temp_c=0.0)   # 0 °C ≈ no diffusion
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl[0][-1, 1] < 0.01

    def test_high_temp_long_step_approaches_full_loss(self):
        model = make_model(n_steps=1, temp_c=1000.0)
        tT = np.array([[1e10, 1000.0]])           # very long step
        grains = make_grains()
        model2 = HeModel(grains_df=grains, tT=tT)
        fl, _ = model2.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl[0][0, 1] > 0.95

    def test_10000_over_T_column_correct(self):
        temp_c = 300.0
        model = make_model(n_steps=3, temp_c=temp_c)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        expected = 1e4 / (temp_c + 273.15)
        assert np.allclose(fl[0][:, 0], expected)

    def test_higher_temp_lower_10000_over_T(self):
        model_lo = make_model(n_steps=2, temp_c=200.0)
        model_hi = make_model(n_steps=2, temp_c=600.0)
        fl_lo, _ = model_lo.step_degas(make_diff_params(), "CN", 1e-6)
        fl_hi, _ = model_hi.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl_hi[0][0, 0] < fl_lo[0][0, 0]


# ── 4. Diffusion type: MP vs CN ───────────────────────────────────────────────

class TestDiffusionType:
    def test_MP_runs_without_error(self):
        model = make_model()
        fl, pr = model.step_degas(make_diff_params(), "MP", 1e-6)
        assert len(fl) == 1

    def test_CN_runs_without_error(self):
        model = make_model()
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 1

    def test_MP_fast_path_profile_nonzero_for_distribute(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params("distribute"), "MP", 1e-6)
        assert np.any(pr[0][:, 1] > 0)

    def test_MP_lattice_profile_nonzero_for_distribute(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params("distribute"), "MP", 1e-6)
        assert np.any(pr[0][:, 2] > 0)

    def test_CN_fast_path_profile_is_zero(self):
        """CN only fills the bulk column; fast/lattice should be zeros."""
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert np.allclose(pr[0][:, 1], 0.0)

    def test_CN_lattice_profile_is_zero(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert np.allclose(pr[0][:, 2], 0.0)


# ── 5. Alpha ejection ─────────────────────────────────────────────────────────

class TestAlphaEjection:
    def test_eject_false_runs(self):
        model = make_model()
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6, eject=False)
        assert len(fl) == 1

    def test_eject_true_runs(self):
        model = make_model()
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6, eject=True)
        assert len(fl) == 1

    def test_no_ejection_raises_total_initial_He(self):
        """Without alpha ejection, more He stays in the grain."""
        model = make_model(n_steps=2, temp_c=200.0)
        fl_ej, _ = model.step_degas(make_diff_params(), "CN", 1e-6, eject=True)
        fl_no, _ = model.step_degas(make_diff_params(), "CN", 1e-6, eject=False)
        # Less He lost when you start with a shallower concentration gradient
        assert fl_no[0][-1, 1] <= fl_ej[0][-1, 1]


# ── 6. Init style ─────────────────────────────────────────────────────────────

class TestInitStyle:
    def test_lattice_style_fast_path_starts_zero(self):
        model = make_model(n_steps=1, temp_c=50.0)   # little diffusion
        _, pr = model.step_degas(make_diff_params("lattice"), "MP", 1e-6)
        # After only one low-T step, fast-path should still be near zero
        assert np.mean(pr[0][:, 1]) < 0.05

    def test_fast_path_style_lattice_starts_zero(self):
        model = make_model(n_steps=1, temp_c=50.0)
        _, pr = model.step_degas(make_diff_params("fast_path"), "MP", 1e-6)
        assert np.mean(pr[0][:, 2]) < 0.05

    def test_distribute_splits_between_pathways(self):
        model = make_model(n_steps=1, temp_c=50.0)
        _, pr = model.step_degas(make_diff_params("distribute"), "MP", 1e-6)
        fast = pr[0][:, 1].sum()
        lat = pr[0][:, 2].sum()
        assert fast > 0 and lat > 0


# ── 7. Damage models (CN path) ────────────────────────────────────────────────

class TestDamageModelsCN:
    @pytest.mark.parametrize("dam_model,diff_model", [
        ("guenthner",  "guenthner"),
        ("guenthner",  "mp_diffusion"),
        ("flowers",    "flowers"),
    ])
    def test_cn_damage_model_runs(self, dam_model, diff_model):
        mineral = "apatite" if dam_model == "flowers" else "zircon"
        model = make_model(mineral=mineral, dam_model=dam_model,
                           diff_model=diff_model)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 1

    def test_unknown_diff_model_returns_none(self):
        model = make_model(diff_model="unknown_model")
        result = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert result is None


# ── 8. Multi-grain behaviour ──────────────────────────────────────────────────

class TestMultiGrain:
    def test_two_grains_return_two_results(self):
        model = make_model(n_grains=2)
        fl, pr = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 2 and len(pr) == 2

    def test_different_grain_sizes_differ(self):
        grains = pd.concat([
            make_grains(size=30.0),
            make_grains(size=120.0),
        ], ignore_index=True)
        model = HeModel(grains_df=grains, tT=make_tT())
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        # Smaller grain should lose more He (shorter diffusion path length)
        assert fl[0][-1, 1] > fl[1][-1, 1]

    def test_mixed_mineral_types(self):
        grains = pd.concat([
            make_grains(mineral="zircon", dam_model="guenthner",
                        diff_model="guenthner"),
            make_grains(mineral="apatite", dam_model="flowers",
                        diff_model="flowers"),
        ], ignore_index=True)
        model = HeModel(grains_df=grains, tT=make_tT())
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert len(fl) == 2


# ── 9. tT recipe edge cases ───────────────────────────────────────────────────

class TestTTRecipe:
    def test_single_step(self):
        model = make_model(n_steps=1, temp_c=300.0)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl[0].shape[0] == 1

    def test_very_short_duration_near_zero_loss(self):
        tT = np.array([[1.0, 300.0]])    # 1 second — negligible diffusion
        grains = make_grains()
        model = HeModel(grains_df=grains, tT=tT)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        assert fl[0][0, 1] < 1e-4

    def test_repeated_identical_steps_monotonic(self):
        model = make_model(n_steps=6, temp_c=400.0)
        fl, _ = model.step_degas(make_diff_params(), "CN", 1e-6)
        vals = fl[0][:, 1]
        assert np.all(np.diff(vals) >= -1e-9)


# ── 10. log2_nodes ────────────────────────────────────────────────────────────

class TestLog2Nodes:
    def test_default_node_count(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "MP", 1e-6)
        assert pr[0].shape[0] == 2**13 + 1

    def test_smaller_node_count(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "MP", 1e-6, log2_nodes=8)
        assert pr[0].shape[0] == 2**8 + 1

    def test_larger_node_count(self):
        model = make_model()
        _, pr = model.step_degas(make_diff_params(), "MP", 1e-6, log2_nodes=14)
        assert pr[0].shape[0] == 2**14 + 1