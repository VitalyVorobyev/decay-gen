""" Tools for calorimeter cluster manipulations

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

from typing import NamedTuple
import numpy as onp

import jax.numpy as np
import jax

from cartesian import Position, Momentum

dtype = np.ndarray

class Cluster(NamedTuple):
    """ Calorimeter cluster """
    energy: dtype  # energy
    costh: dtype  # cosine of the polar angle
    phi: dtype  # azimuth angle

    @staticmethod
    def from_ndarray(data: np.ndarray):
        """ """
        assert data.shape[1] == 3
        return Cluster(*[data[:,i] for i in range(3)])

    @property
    def as_array(self) -> (np.ndarray):
        return np.column_stack([self.energy, self.costh, self.phi])

    @property
    def sinth(self):
        """ Sine of the polar angle """
        return np.sqrt(1 - self.costh**2)


def cartesian_to_cluster(pos: Position, mom: Momentum, R=1000) -> (Cluster):
    """ Make cluster from Position and Momentum """
    # Find intersection with calorimeter (see the geometry sketch)
    pos_mom_scalar_product = pos.x*mom.px + pos.y*mom.py + pos.z*mom.pz
    mom_total = mom.ptot
    mom_total_squared = mom_total**2
    
    full_computation = False
    if full_computation:
        alpha = (-pos_mom_scalar_product + np.sqrt(
            pos_mom_scalar_product**2 + mom_total_squared * (R**2 - pos.r**2)
        )) / mom_total_squared
    else:  # takes into account that R >> particle flight length
        alpha = R / mom_total - pos_mom_scalar_product / mom_total_squared
    
    cluster_position = Position.from_ndarray(pos.as_array + alpha * mom.as_array)

    return Cluster.from_ndarray(np.column_stack([
        mom_total, cluster_position.costh, cluster_position.phi
    ]))


def momentum_from_cluster(clu: Cluster) -> (Momentum):
    """ Assuming photon (massless particle) """
    sinth = clu.sinth
    return Momentum.from_ndarray(np.column_stack([
        clu.energy * sinth * np.sin(clu.phi),
        clu.energy * sinth * np.cos(clu.phi),
        clu.energy * clu.costh,
    ]))

momentum_from_cluster_jacobian = jax.vmap(jax.jacfwd(momentum_from_cluster))

@jax.vmap
def cluster_covariance(clu: Cluster) -> (dtype):
    """ [energy, costh, phi] """
    eps = 5.e-2
    return np.diag(np.array([
        clu.energy, clu.costh, clu.phi
    ]))**2 * eps**2

@jax.vmap
def sample_cluster_resolution(clu: Cluster) -> (Cluster):
    """ Sample helix parameters given true parameters and covariance matrix """
    rng = jax.random.PRNGKey(seed=0)
    cov = cluster_covariance(clu)
    dclu = jax.random.multivariate_normal(rng, np.zeros((cov.shape[-1])), cov)
    return Cluster.from_ndarray(clu.as_array + dclu)
