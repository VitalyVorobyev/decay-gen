""" Tools for track helix manipulations

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: August 2020
"""

from typing import NamedTuple

import jax.numpy as np
import jax

from .cartesian import Position, Momentum, alpha

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
        """ Factory method that builds Helix from np.nparray """
        assert data.shape[1] == 5
        return Helix(*[data[:, i] for i in range(5)])

    @property
    def as_array(self) -> (np.ndarray):
        """ Helix parameters as np.ndarray """
        return np.column_stack([
            self.d0, self.phi0, self.omega, self.z0, self.tanl])

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

    def phi(self, length: dtype) -> (dtype):
        """ length: flight length """
        return self.phi0 + self.omega * length


def position_from_helix(hel: Helix, length: dtype, q: dtype, B: float)\
        -> (Position):
    """ Construct Position from Helix """
    r, phi = hel.r, hel.phi(length)
    return Position(
        x=r*np.sin(phi) - (r+hel.d0)*np.sin(hel.phi0),
        y=-r*np.cos(phi) + (r+hel.d0)*np.cos(hel.phi0),
        z=hel.z0 + length * hel.tanl)


def momentum_from_helix(hel: Helix, length: dtype, q: dtype, B: float)\
        -> (Momentum):
    """ Construct Momentum from Helix """
    phi, pt = hel.phi(length), hel.pt(q, B)
    return Momentum(
        px=pt*np.cos(phi),
        py=pt*np.sin(phi),
        pz=pt*hel.tanl)


def helix_to_cartesian(hel: Helix, length: dtype, q: dtype, B: float)\
        -> (Position, Momentum):
    """ Helper function to construct Vertex and Momentum from Helix """
    return (
        position_from_helix(hel, length, q, B),
        momentum_from_helix(hel, length, q, B))


def cartesian_to_helix(pos: Position, mom: Momentum, q: dtype, B: float)\
        -> (Helix, dtype):
    """ Construct Helix from Momentum and Position """
    qalph = q * alpha(B)
    pt = mom.pt
    phi = np.arctan2(mom.py, mom.px)
    phi0, pt0 = mom.phi0pt0(pos, q, B)
    # The flight length in the transverse plane, measured from the point of
    # the helix closeset to the z-axis
    length = (phi - phi0) * pt / qalph
    return (
        Helix(
            d0=(pt0 - pt) / qalph,
            phi0=phi0,
            omega=qalph / pt,
            z0=pos.z - length * mom.pz / pt,
            tanl=mom.pz/pt),
        length)


position_from_helix_jacobian = jax.vmap(
    jax.jacfwd(position_from_helix, argnums=0))


momentum_from_helix_jacobian = jax.vmap(
    jax.jacfwd(momentum_from_helix, argnums=0))


def full_jacobian_from_helix(hel: Helix, length: dtype, q: int, B: float)\
        -> (dtype):
    """ Calculates helix over (pos, mom) jacobian.
    Returns np.array of shape (N, 5, 6), where N is number of events """
    jac_pos = position_from_helix_jacobian(hel, length, q, B)
    jac_mom = momentum_from_helix_jacobian(hel, length, q, B)

    return np.stack([
        jac_pos.x.as_array,
        jac_pos.y.as_array,
        jac_pos.z.as_array,
        jac_mom.px.as_array,
        jac_mom.py.as_array,
        jac_mom.pz.as_array,
    ], axis=2)
