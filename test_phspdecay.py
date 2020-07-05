""" Unit tests for the phspdecay package """

import phspdecay as d
import numpy as np

def test_pname():
    for name in ['pi+', 'pi-', 'K+', 'K-', 'gamma', 'pi0', 'K(S)0',\
        'K(L)0', 'D0', 'D+', 'K*(892)+', 'K*(892)-', 'K*(892)0']:
        assert d.pname(name, name).name == name


def test_generate():
    N = 100
    w, p, _ = d.generate('', N)

    assert w.shape == (N, )
    assert p['K-'].shape == (N, 4)
    assert np.all(p['K-'][:,-1] > 0)  # energy is the last


def test_genmom_to_helix():
    N = 100
    _, p, _ = d.generate('', N)

    pos, hel, l = d.genmom_to_helix(p['pi+'].numpy())

    assert l.shape == (N, )
    assert hel.as_array.shape == (N, 5)
    assert pos.as_array.shape == (N, 3)
