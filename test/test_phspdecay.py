""" Unit tests for the phspdecay package """

from helixgen.phspdecay import pname, generate
import jax.random as rjax

rng = rjax.PRNGKey(seed=0)

def test_pname():
    for name in ['pi+', 'pi-', 'K+', 'K-', 'gamma', 'pi0', 'K(S)0',\
        'K(L)0', 'D0', 'D+', 'K*(892)+', 'K*(892)-', 'K*(892)0']:
        gpcl, pcl = pname(name, name)
        assert pcl.name == name
        assert gpcl.name == name


def test_generate():
    N = 100
    w, genpcls = generate(rng, '', N)
    key = 'pi+'
    assert w.shape == (N, )
    assert genpcls[key]['mom'].as_array.shape == (N, 3)
    assert genpcls[key]['pos'].as_array.shape == (N, 3)
    assert genpcls[key]['hel'].as_array.shape == (N, 5)
    assert genpcls[key]['meas_pos'].as_array.shape == (N, 3)
    assert genpcls[key]['meas_mom'].as_array.shape == (N, 3)
    assert genpcls[key]['helcov'].shape == (N, 5, 5)
