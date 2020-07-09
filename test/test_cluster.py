""" Unit tests for the cartesian package """

import pytest
from helixgen.cartesian import Position, Momentum
from helixgen.cluster import Cluster, momentum_from_cluster_jacobian, cartesian_to_cluster,\
    momentum_from_cluster, cluster_covariance, sample_cluster_resolution
import jax.random as rjax
import jax.numpy as np

rng = rjax.PRNGKey(seed=0)

def random_cluster(rng, N):
    """ Helper function """
    energy = rjax.uniform(rng, (N,), minval=0., maxval=3.)
    costh = rjax.uniform(rng, (N,), minval=-1., maxval=1.)
    phi = rjax.uniform(rng, (N,), minval=-np.pi, maxval=np.pi)
    return Cluster(energy, costh, phi)

def test_cluster_ctor():
    """ """
    energy = rjax.uniform(rng, minval=0., maxval=3.)
    costh = rjax.uniform(rng, minval=-1., maxval=1.)
    phi = rjax.uniform(rng, minval=-np.pi, maxval=np.pi)
    clu = Cluster(energy, costh, phi)
    assert clu.energy == energy
    assert clu.costh == costh
    assert clu.phi == phi


def test_cluster_ctor_arrays():
    """ """
    N = 100
    energy = rjax.uniform(rng, (N,), minval=0., maxval=3.)
    costh = rjax.uniform(rng, (N,), minval=-1., maxval=1.)
    phi = rjax.uniform(rng, (N,), minval=-np.pi, maxval=np.pi)
    clu = Cluster(energy, costh, phi)
    assert np.allclose(clu.energy, energy)
    assert np.allclose(clu.costh, costh)
    assert np.allclose(clu.phi, phi)
    assert clu.as_array.shape == (N, 3)


def test_cluster_from_ndarray():
    """ """
    N = 100
    energy = rjax.uniform(rng, (N,), minval=0., maxval=3.)
    costh = rjax.uniform(rng, (N,), minval=-1., maxval=1.)
    phi = rjax.uniform(rng, (N,), minval=-np.pi, maxval=np.pi)
    clu = Cluster.from_ndarray(np.column_stack([energy, costh, phi]))
    assert np.allclose(clu.energy, energy)
    assert np.allclose(clu.costh, costh)
    assert np.allclose(clu.phi, phi)
    assert clu.as_array.shape == (N, 3)


def test_momentum_from_cluster_jacobian():
    """ """
    N = 100
    clu = random_cluster(rng, N)
    jac = momentum_from_cluster_jacobian(clu)

    assert jac.px.energy.shape == (N, 1)  # d(px) / d(energy) or d(energy) / d(px) ?
    assert jac.py.energy.shape == (N, 1)
    assert jac.pz.energy.shape == (N, 1)
    assert jac.px.costh.shape == (N, 1)
    assert jac.py.costh.shape == (N, 1)
    assert jac.pz.costh.shape == (N, 1)
    assert jac.px.phi.shape == (N, 1)
    assert jac.py.phi.shape == (N, 1)
    assert jac.pz.phi.shape == (N, 1)


def test_cartesian_to_cluster():
    """ """
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    clu = cartesian_to_cluster(pos, mom)

    assert clu.as_array.shape == (N, 3)


def test_momentum_from_cluster():
    """ """
    N = 100
    clu = random_cluster(rng, N)
    mom = momentum_from_cluster(clu)
    
    assert mom.as_array.shape == (N, 3)


def test_cluster_covariance():
    """ """
    N = 100
    clu = random_cluster(rng, N)
    cov = cluster_covariance(clu)

    assert cov.shape == (N, 3, 3)


def test_sample_cluster_resolution():
    N = 100
    clu = random_cluster(rng, N)
    sclu, cov = sample_cluster_resolution(rng, clu)

    assert cov.shape == (N, 3, 3)
    assert sclu.as_array.shape == (N, 3)
