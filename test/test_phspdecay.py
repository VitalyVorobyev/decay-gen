""" Unit tests for the phspdecay package """

from helixgen.phspdecay import pname, generate
import numpy as np

def test_pname():
    for name in ['pi+', 'pi-', 'K+', 'K-', 'gamma', 'pi0', 'K(S)0',\
        'K(L)0', 'D0', 'D+', 'K*(892)+', 'K*(892)-', 'K*(892)0']:
        gpcl, pcl = pname(name, name)
        assert pcl.name == name
        assert gpcl.name == name


def test_generate():
    N = 100
    w, genpcls = generate('', N)

    assert w.shape == (N, )
    assert genpcls['K-']['mom'].as_array.shape == (N, 3)
    assert genpcls['K-']['pos'].as_array.shape == (N, 3)
    assert genpcls['K-']['hel'].as_array.shape == (N, 5)
    assert genpcls['K-']['meas_pos'].as_array.shape == (N, 3)
    assert genpcls['K-']['meas_mom'].as_array.shape == (N, 3)
    assert genpcls['K-']['helcov'].shape == (N, 5, 5)
