""" Tools for resolution smearing

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: August 2020
    Modified: August 2020
"""

from typing import NamedTuple, Callable
import numpy as onp
import jax.numpy as np
import jax

from particle import Particle

from .helix import Helix, cartesian_to_helix, helix_to_cartesian
from .cluster import Cluster, cartesian_to_cluster, momentum_from_cluster
from .cartesian import Position, Momentum

dtype = np.ndarray

@jax.vmap
def position_covariance(pos: Position) -> (dtype):
    """ Covariance matrix for spatial position """
    epssq = 2.e-2**2
    return np.diag(np.array([pos.x, pos.y, pos.z])**2) * epssq +\
        np.diag(np.array([0.03, 0.03, 0.05])**2)

@jax.vmap
def momentum_covariance(mom: Momentum) -> (dtype):
    """ Covariance matrix for momentum """
    epssq = 1.e-2**2
    return np.diag(np.array([mom.px, mom.py, mom.pz])**2) * epssq +\
        np.diag(np.array([10, 10, 17])**2)

@jax.vmap
def helix_covariance(hel: Helix) -> (dtype):
    """ [d0, phi0 omega, z0, tan(lambda)] """
    epssq = 2.e-2**2
    return np.diag(np.array(
        [hel.d0, hel.phi0, 0.1*hel.z0, hel.omega, hel.tanl])**2) * epssq +\
            np.diag(np.array([0.08, 0.01, 0.001, 0.01, 0.001])**2)

@jax.vmap
def cluster_covariance(clu: Cluster) -> (dtype):
    """ [energy, costh, phi] """
    epssq = 5.e-2**2
    return np.diag(np.array([clu.energy, clu.costh, clu.phi])**2) * epssq +\
        np.diag(np.array([10, 0.01, 0.03])**2)


def sample_resolution(rng: jax.random.PRNGKey, obj: NamedTuple, covgen: Callable)\
        -> (NamedTuple, dtype):
    """ Resolution sampler
        TODO: find out how to vectorize multivariate_normal
    """
    cov = covgen(obj)

    objarr = obj.as_array
    newobj = onp.empty(objarr.shape)

    for i, [h, c] in enumerate(zip(objarr, cov)):
        newobj[i,:] = h + onp.random.multivariate_normal(onp.zeros(c.shape[-1]), c)

    return (type(obj).from_ndarray(newobj), cov)
    # mvn = jax.vmap(lambda c: jax.random.multivariate_normal(rng, np.zeros(c.shape[-1]), c))
    # return (Helix.from_ndarray(hel.as_array + mvn(cov)), cov)


def sample_helix_resolution(rng: jax.random.PRNGKey, hel: Helix) -> (Helix, dtype):
    """ Helix resolution sampler """
    return sample_resolution(rng, hel, helix_covariance)


def sample_position_resolution(rng: jax.random.PRNGKey, pos: Position) -> (Position, dtype):
    """ Position resolution sampler """
    return sample_resolution(rng, pos, position_covariance)


def sample_momentum_resolution(rng: jax.random.PRNGKey, mom: Momentum) -> (Momentum, dtype):
    """ Momentum resolution sampler """
    return sample_resolution(rng, mom, momentum_covariance)


def sample_cluster_resolution(rng: jax.random.PRNGKey, clu: Cluster) -> (Cluster, np.ndarray):
    """ Sample cluster parameters given the true values """
    return sample_resolution(rng, clu, cluster_covariance)


def apply_resolution_charged_cartesian(
        rng: jax.random.PRNGKey, pos: Position, mom: Momentum, pcl: Particle)\
        -> (Helix, np.ndarray, Position, Momentum):
    """ Smear track using covariance of the cartesian parameters """
    q, B = pcl.charge, 1.5
    pos_smeared, poscov = sample_position_resolution(rng, pos)
    mom_smeared, momcov = sample_momentum_resolution(rng, mom)
    return {
        'meas_pos': pos_smeared,
        'meas_mom': mom_smeared,
        'poscov': poscov,
        'momcov': momcov,
        'meas_hel': cartesian_to_helix(pos_smeared, mom_smeared, q=q, B=B)[0],
        'hel': cartesian_to_helix(pos, mom, q=q, B=B)[0],
    }


def apply_resolution_charged(rng: jax.random.PRNGKey, pos: Position, mom: Momentum, pcl: Particle)\
    -> (Helix, np.ndarray, Position, Momentum):
    """ returns measured helix, helix covariance, position and momentum """
    q, B = pcl.charge, 1.5
    hel, _ = cartesian_to_helix(pos, mom, q=q, B=B)
    hel_smeared, helcov = sample_helix_resolution(rng, hel)
    pos, mom = helix_to_cartesian(hel, l=0., q=q, B=B)
    return {
        'hel': hel,
        'meas_hel': hel_smeared,
        'meas_pos': pos,
        'meas_mom': mom,
        'helcov': helcov
    }


def apply_resolution_neutral_cartesian(rng: jax.random.PRNGKey, pos: Position, mom: Momentum)\
    -> (Cluster, np.ndarray, Momentum):
    """ Smear track using covariance of the cartesian parameters """
    pos_smeared, poscov = sample_position_resolution(rng, pos)
    mom_smeared, momcov = sample_momentum_resolution(rng, mom)
    return {
        'clu': cartesian_to_cluster(pos, mom),
        'meas_clu': cartesian_to_cluster(pos_smeared, mom_smeared),
        'meas_pos': pos_smeared,
        'meas_mom': mom_smeared,
        'poscov': poscov,
        'momcov': momcov
    }


def apply_resolution_neutral(rng: jax.random.PRNGKey, pos: Position, mom: Momentum)\
    -> (Cluster, np.ndarray, Momentum):
    """ returns measured cluster, cluster covariance, and momentum """
    clu = cartesian_to_cluster(pos, mom)
    clu_smeared, clucov = sample_cluster_resolution(rng, clu)
    return {
        'clu': clu,
        'meas_clu': clu_smeared,
        'clucov': clucov,
        'meas_mom': momentum_from_cluster(clu_smeared)
    }


stable_charged = set(['pi+', 'pi-', 'e+', 'e-', 'p', 'anti-p', 'K+', 'K-', 'mu+', 'mu-'])
stable_neutral = set(['gamma'])

def apply_resolution(rng: jax.random.PRNGKey, genpcls: dict,
                     smearer_neutrals: Callable, smearer_tracks: Callable) -> (dict):
    for name, data in genpcls.items():
        if data['pcl'].name in stable_neutral:
            genpcls[name].update(smearer_neutrals(rng, data['pos'], data['mom']))
        elif data['pcl'].name in stable_charged:
            genpcls[name].update(smearer_tracks(rng, data['pos'], data['mom'], data['pcl']))
    return genpcls


def apply_resolution_helclu(rng: jax.random.PRNGKey, genpcls: dict) -> (dict):
    return apply_resolution(rng, genpcls,
                            apply_resolution_neutral,
                            apply_resolution_charged)


def apply_resolution_cartesian(rng: jax.random.PRNGKey, genpcls: dict) -> (dict):
    return apply_resolution(rng, genpcls,
                            apply_resolution_neutral_cartesian,
                            apply_resolution_charged_cartesian)
