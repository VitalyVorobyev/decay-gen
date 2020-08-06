""" Unit tests for the resolution module """

from helixgen.helix import Helix
from helixgen.cartesian import Position, Momentum
from helixgen.cluster import Cluster
from helixgen.resolution import *

from particle import Particle

import jax.random as rjax
import jax.numpy as np

rng = rjax.PRNGKey(seed=0)

def test_position_covariance():
    """ """
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    cov = position_covariance(pos)

    assert cov.shape == (N, 3, 3)
    assert not np.isnan(cov).any()

def test_momentum_covariance():
    """ """
    N = 100
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    cov = momentum_covariance(mom)

    assert cov.shape == (N, 3, 3)
    assert not np.isnan(cov).any()

def test_helix_covariance():
    """ """
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    cov = helix_covariance(hel)

    assert cov.shape == (N, 5, 5)
    assert not np.isnan(cov).any()

def test_cluster_covariance():
    """ """
    N = 100
    clu = Cluster.from_ndarray(rjax.uniform(rng, (N, 3)))
    cov = cluster_covariance(clu)

    assert cov.shape == (N, 3, 3)
    assert not np.isnan(cov).any()

def test_sample_helix_resolution():
    N = 100
    hel = Helix.from_ndarray(rjax.uniform(rng, (N, 5)))
    shel, cov = sample_helix_resolution(rng, hel)

    assert cov.shape == (N, 5, 5)
    assert shel.as_array.shape == (N, 5)
    assert not np.isnan(cov).any()
    assert not np.isnan(shel.as_array).any()

def test_sample_position_resolution():
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    spos, cov = sample_position_resolution(rng, pos)

    assert cov.shape == (N, 3, 3)
    assert spos.as_array.shape == (N, 3)
    assert not np.isnan(cov).any()
    assert not np.isnan(spos.as_array).any()

def test_sample_momentum_resolution():
    N = 100
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    smom, cov = sample_momentum_resolution(rng, mom)

    assert cov.shape == (N, 3, 3)
    assert smom.as_array.shape == (N, 3)
    assert not np.isnan(cov).any()
    assert not np.isnan(smom.as_array).any()


def test_sample_cluster_resolution():
    N = 100
    clu = Cluster.from_ndarray(rjax.uniform(rng, (N, 3)))
    sclu, cov = sample_momentum_resolution(rng, clu)

    assert cov.shape == (N, 3, 3)
    assert sclu.as_array.shape == (N, 3)
    assert not np.isnan(cov).any()
    assert not np.isnan(sclu.as_array).any()

# def test_apply_resolution_charged_cartesian():
#     pass
