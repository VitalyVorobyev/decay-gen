""" Unit tests for the phspdecay package """

from helixgen.phspdecay import pname, generate
from helixgen.resolution import apply_resolution
from helixgen.resolution import apply_resolution_neutral, apply_resolution_neutral_cartesian
from helixgen.resolution import apply_resolution_charged, apply_resolution_charged_cartesian

import jax.random as rjax
rng = rjax.PRNGKey(seed=0)

def test_pname():
    for name in ['pi+', 'pi-', 'K+', 'K-', 'gamma', 'pi0', 'K(S)0',\
        'K(L)0', 'D0', 'D+', 'K*(892)+', 'K*(892)-', 'K*(892)0']:
        gpcl, pcl = pname(name, name)
        assert pcl.name == name
        assert gpcl.name == name


def test_generate_helix():
    N = 100
    smerer = lambda rng, gp: apply_resolution(
        rng, gp,
        apply_resolution_charged,
        apply_resolution_charged)
    w, genpcls = generate(rng, '', N, smerer)
    key = 'pi+'
    assert w.shape == (N, )
    assert genpcls[key]['mom'].as_array.shape == (N, 3)
    assert genpcls[key]['pos'].as_array.shape == (N, 3)
    assert genpcls[key]['hel'].as_array.shape == (N, 5)
    assert genpcls[key]['meas_pos'].as_array.shape == (N, 3)
    assert genpcls[key]['meas_mom'].as_array.shape == (N, 3)
    assert genpcls[key]['meas_hel'].as_array.shape == (N, 5)
    assert genpcls[key]['helcov'].shape == (N, 5, 5)

def test_generate_cartesian():
    N = 100
    smerer = lambda rng, gp: apply_resolution(
        rng, gp,
        apply_resolution_charged_cartesian,
        apply_resolution_charged_cartesian)
    w, genpcls = generate(rng, '', N, smerer)
    key = 'pi+'
    assert w.shape == (N, )
    assert genpcls[key]['mom'].as_array.shape == (N, 3)
    assert genpcls[key]['pos'].as_array.shape == (N, 3)
    assert genpcls[key]['hel'].as_array.shape == (N, 5)
    assert genpcls[key]['meas_pos'].as_array.shape == (N, 3)
    assert genpcls[key]['meas_mom'].as_array.shape == (N, 3)
    assert genpcls[key]['meas_hel'].as_array.shape == (N, 5)
    assert genpcls[key]['poscov'].shape == (N, 3, 3)
    assert genpcls[key]['momcov'].shape == (N, 3, 3)
