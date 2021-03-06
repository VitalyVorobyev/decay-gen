""" Unit tests for the helix module """

import pytest
from helixgen.cartesian import Position, Momentum
from helixgen.helix import Helix, position_from_helix, momentum_from_helix, helix_to_cartesian,\
    cartesian_to_helix, position_from_helix_jacobian, momentum_from_helix_jacobian,\
        full_jacobian_from_helix
import jax.random as rjax
import jax.numpy as np

rng = rjax.PRNGKey(seed=0)

def test_helix_ctor():
    """ """
    d0, phi0, omega, z0, tanl = rjax.uniform(rng, (5,))
    hel = Helix(d0, phi0, omega, z0, tanl)
    assert hel.d0 == d0
    assert hel.phi0 == phi0
    assert hel.omega == omega
    assert hel.z0 == z0
    assert hel.tanl == tanl


def test_helix_ctor_named():
    """ """
    d0, phi0, omega, z0, tanl = rjax.uniform(rng, (5,))
    hel = Helix(d0=d0, phi0=phi0, omega=omega, z0=z0, tanl=tanl)
    assert hel.d0 == d0
    assert hel.phi0 == phi0
    assert hel.omega == omega
    assert hel.z0 == z0
    assert hel.tanl == tanl


def test_helix_ctor_named_reorder():
    """ """
    d0, phi0, omega, z0, tanl = rjax.uniform(rng, (5,))
    hel = Helix(phi0=phi0, omega=omega, tanl=tanl, z0=z0, d0=d0)
    assert hel.d0 == d0
    assert hel.phi0 == phi0
    assert hel.omega == omega
    assert hel.z0 == z0
    assert hel.tanl == tanl


def test_helix_ctor_ndarray():
    """ """
    N = 100
    d0, phi0, omega, z0, tanl = [rjax.uniform(rng, (N,)) for i in range(5)]
    hel = Helix(d0, phi0, omega, z0, tanl)

    assert np.allclose(d0, hel.d0)
    assert np.allclose(phi0, hel.phi0)
    assert np.allclose(omega, hel.omega)
    assert np.allclose(z0, hel.z0)
    assert np.allclose(tanl, hel.tanl)


def test_helix_from_ndarray():
    N = 100
    data = rjax.uniform(rng, (N, 5))
    d0, phi0, omega, z0, tanl = [data[:,i] for i in range(5)]

    hel = Helix.from_ndarray(data)

    assert np.allclose(d0, hel.d0)
    assert np.allclose(phi0, hel.phi0)
    assert np.allclose(omega, hel.omega)
    assert np.allclose(z0, hel.z0)
    assert np.allclose(tanl, hel.tanl)


def test_helix_as_ndarray():
    N = 100
    data = rjax.uniform(rng, (N, 5))
    hel = Helix.from_ndarray(data)
    assert np.allclose(data, hel.as_array)


def test_position_from_helix():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.

    pos = position_from_helix(hel, l, q, B)
    assert pos.x.shape == (N,)
    assert pos.y.shape == (N,)
    assert pos.z.shape == (N,)


def test_momentum_from_helix():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.5

    p = momentum_from_helix(hel, l, q, B)
    assert p.px.shape == (N,)
    assert p.py.shape == (N,)
    assert p.pz.shape == (N,)


@pytest.mark.skip(reason='ambiguity of the helix parameters')
def test_helix_to_cartesian():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.5

    pos, mom = helix_to_cartesian(hel, l, q, B)
    hel2, l2 = cartesian_to_helix(pos, mom, q, B)

    assert np.allclose(l, l2)
    assert np.allclose(hel.d0, hel2.d0)
    assert np.allclose(hel.phi0, hel2.phi0)
    assert np.allclose(hel.omega, hel2.omega)
    assert np.allclose(hel.z0, hel2.z0)
    assert np.allclose(hel.tanl, hel2.tanl)
    assert np.allclose(hel.pt(q, B), hel2.pt(q, B))
    assert np.allclose(hel.as_array, hel2.as_array)


def test_cartesian_to_helix():
    """ """
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    q = rjax.choice(rng, [-1, 1], (N,))
    B = 1.5

    hel, l = cartesian_to_helix(pos, mom, q, B)
    pos2, mom2 = helix_to_cartesian(hel, l, q, B)

    rtol, atol = 1.e-4, 1.e-4

    assert np.allclose(pos.x, pos2.x, rtol=rtol, atol=atol)
    assert np.allclose(pos.y, pos2.y, rtol=rtol, atol=atol)
    assert np.allclose(pos.z, pos2.z, rtol=rtol, atol=atol)
    assert np.allclose(pos.as_array, pos2.as_array, rtol=rtol, atol=atol)
    assert np.allclose(mom.px, mom2.px, rtol=rtol, atol=atol)
    assert np.allclose(mom.py, mom2.py, rtol=rtol, atol=atol)
    assert np.allclose(mom.pz, mom2.pz, rtol=rtol, atol=atol)
    assert np.allclose(mom.as_array, mom2.as_array, rtol=rtol, atol=atol)


@pytest.mark.skip(reason='ambiguity of the helix parameters')
def test_double_cartesian_to_helix():
    """ """
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    q = rjax.choice(rng, [-1, 1], (N,))
    B = 1.5

    hel, l = cartesian_to_helix(pos, mom, q, B)
    pos2, mom2 = helix_to_cartesian(hel, l, q, B)
    hel2, l2 = cartesian_to_helix(pos2, mom2, q, B)
    pos3, mom3 = helix_to_cartesian(hel2, l2, q, B) 

    assert np.allclose(pos.x, pos3.x)
    assert np.allclose(pos.y, pos3.y)
    assert np.allclose(pos.z, pos3.z)
    assert np.allclose(pos.as_array, pos3.as_array)
    assert np.allclose(mom.px, mom3.px)
    assert np.allclose(mom.py, mom3.py)
    assert np.allclose(mom.pz, mom3.pz)
    assert np.allclose(mom.as_array, mom3.as_array)


def test_position_from_helix_jacobian():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.5 * np.ones(N)

    jac = position_from_helix_jacobian(hel, l, q, B)

    assert jac.x.as_array.shape == (N, 5)
    assert jac.y.as_array.shape == (N, 5)
    assert jac.z.as_array.shape == (N, 5)


def test_momentum_from_helix_jacobian():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.5 * np.ones(N)

    jac = momentum_from_helix_jacobian(hel, l, q, B)

    assert jac.px.as_array.shape == (N, 5)
    assert jac.py.as_array.shape == (N, 5)
    assert jac.pz.as_array.shape == (N, 5)

def test_full_jacobian_from_helix():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    q = rjax.choice(rng, [-1, 1], (N,))
    l = rjax.uniform(rng, (N,))
    B = 1.5 * np.ones(N)

    jac = full_jacobian_from_helix(hel, l, q, B)

    assert jac.shape == (N, 5, 6)
    assert not np.isnan(jac).any()
