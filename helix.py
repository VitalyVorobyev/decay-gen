""" Tools for track helix manipulations

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

from typing import NamedTuple

import jax.numpy as np
import jax

from cartesian import Position, Momentum, alpha

dtype = np.ndarray

class Helix(NamedTuple):
    """ Helix representation as in W. Hulsbergen NIM 552 (2005) 566  """
    d0: dtype
    phi0: dtype
    omega: dtype
    z0: dtype
    tanl: dtype

    @staticmethod
    def from_ndarray(data: np.ndarray):
        """ """
        assert data.shape[1] == 5
        return Helix(*[data[:,i] for i in range(5)])

    @property
    def as_array(self) -> (np.ndarray):
        return np.column_stack([
            self.d0, self.phi0, self.omega, self.z0, self.tanl
        ])

    def pt(self, q: dtype, B: float) -> (dtype):
        """ Transverce momentum given particle charge and magnetic field
        Args:
            - q: particle charge [units of the positron charge]
            - B: magnetic field [T]
        """
        return q * alpha(B) / self.omega

    @property
    def r(self) -> (dtype):
        """ Radius of curvature """
        return 1. / self.omega


    def phi(self, l: dtype) -> (dtype):
        """ l: flight length """
        return self.phi0 + self.omega * l


def position_from_helix(hel: Helix, l: dtype, q: dtype, B: float) -> (Position):
    """ Construct Position from Helix """
    r, phi = hel.r, hel.phi(l)
    return Position(
        x= r*np.sin(phi) - (r+hel.d0)*np.sin(hel.phi0),
        y=-r*np.cos(phi) + (r+hel.d0)*np.cos(hel.phi0),
        z=hel.z0 + l*hel.tanl
    )

def momentum_from_helix(hel: Helix, l: dtype, q: dtype, B: float) -> (Momentum):
    """ Construct Momentum from Helix """
    phi, pt = hel.phi(l), hel.pt(q, B)
    return Momentum(
        px=pt*np.cos(phi),
        py=pt*np.sin(phi),
        pz=pt*hel.tanl
    )


def helix_to_cartesian(hel: Helix, l: dtype, q: dtype, B: float)\
    -> (Position, Momentum):
    """ Helper function to construct Vertex and Momentum from Helix """
    return (
        position_from_helix(hel, l, q, B),
        momentum_from_helix(hel, l, q, B)
    )


def cartesian_to_helix(pos: Position, mom: Momentum, q: dtype, B: float)\
    -> (Helix, dtype):
    """ Construct Helix from Momentum and Position """
    qalph = q * alpha(B)
    pt = mom.pt
    phi = np.arctan2(mom.py, mom.px)
    phi0, pt0 = mom.phi0pt0(pos, q, B)
    l = (phi - phi0) * pt / qalph  # The flight length in the transverse plane, measured
                                   # from the point of the helix closeset to the z-axis
    return (Helix(
        d0=(pt0 - pt) / qalph,
        phi0=phi0,
        omega=qalph / pt,
        z0=pos.z - l * mom.pz / pt,
        tanl=mom.pz/pt
    ), l)


position_from_helix_jacobian = jax.vmap(jax.jacfwd(position_from_helix, argnums=0))
momentum_from_helix_jacobian = jax.vmap(jax.jacfwd(momentum_from_helix, argnums=0))

def full_jacobian_from_helix(hel: Helix, l: dtype, q: int, B: float) -> (dtype):
    """ Calculates helix over (pos, mom) jacobian.
    Returns np.array of shape (N, 5, 6), where N is number of events """
    jac_pos = position_from_helix_jacobian(hel, l, q, B)
    jac_mom = momentum_from_helix_jacobian(hel, l, q, B)

    return np.stack([
        jac_pos.x.as_array,
        jac_pos.y.as_array,
        jac_pos.z.as_array,
        jac_mom.px.as_array,
        jac_mom.py.as_array,
        jac_mom.pz.as_array,
    ], axis=2)


@jax.vmap
def helix_covariance(hel: Helix) -> (dtype):
    """ [d0, phi0 omega, z0, tan(lambda)] """
    eps = 5.e-2
    return np.diag(np.array([
        hel.d0, hel.phi0, hel.z0, hel.omega, hel.tanl
    ]))**2 * eps**2


@jax.vmap
def sample_helix_resolution(hel: Helix) -> (Helix):
    """ Sample helix parameters given true parameters and covariance matrix """
    rng = jax.random.PRNGKey(seed=0)
    cov = helix_covariance(hel)
    dhel = jax.random.multivariate_normal(rng, np.zeros((cov.shape[-1])), cov)
    return (Helix.from_ndarray(hel.as_array + dhel), cov)


# helix_covariance = jax.vmap(helix_covariance_flat)
# sample_helix_resolution = jax.vmap(sample_helix_resolution_flat)
