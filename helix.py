""" Tools for track helix manipulations

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

from typing import NamedTuple
import numpy as onp

import jax.numpy as np
import jax
    
dtype = np.ndarray

speedOfLight = 29.9792458  # [cm / ns]

def alpha(B):
    """ """
    return 1.0 / (B * speedOfLight) * 1e4


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


class Position(NamedTuple):
    """ Position on helix """
    x: dtype
    y: dtype
    z: dtype

    @staticmethod
    def from_ndarray(data: np.ndarray):
        """ """
        assert data.shape[1] == 3
        return Position(*[data[:,i] for i in range(3)])


    @property
    def as_array(self) -> (np.ndarray):
        return np.column_stack([self.x, self.y, self.z])


    def __sub__(self, rhs):
        """ Position subtraction """
        return Position.from_ndarray(self.as_array - rhs.as_array)


    def __div__(self, coef: float):
        """ Division by float """
        return Position.from_ndarray(self.as_array / coef)


    def __mul__(self, coef: float):
        """ Multiplication by float """
        return Position.from_ndarray(self.as_array * coef)

    
class Momentum(NamedTuple):
    """ Particle momentum """
    px: dtype
    py: dtype
    pz: dtype

    @staticmethod
    def from_ndarray(data: np.ndarray):
        """ """
        assert data.shape[1] == 3
        return Momentum(*[data[:,i] for i in range(3)])


    @property
    def as_array(self) -> (np.ndarray):
        return np.column_stack([self.px, self.py, self.pz])

    
    @property
    def pt(self) -> (dtype):
        """ Transverse momentum """
        return np.sqrt(self.px**2 + self.py**2)


    def px0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ """
        return self.px + pos.y * q * alpha(B)


    def py0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ """
        return self.py - pos.x * q * alpha(B)


    def pt0(self, pos: Position, q: dtype, B: float)\
        -> (dtype):
        """ """
        return np.sqrt(self.px0(pos, q, B)**2 +
                       self.py0(pos, q, B)**2)


    def phi0pt0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ Helper function that calculates phi0 and pt0 efficiently """
        px0_ = self.px0(pos, q, B)
        py0_ = self.py0(pos, q, B)
        return (np.arctan2(py0_, px0_), np.sqrt(px0_**2 + py0_**2))


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

def jacobian(hel, l, q, B):
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


def helix_covariance(hel):
    """ """
    pass


def cartesian_covariance(hel):
    """ """
    pass


def sample_helix_resolution(hel):
    """ """
    pass


if __name__ == '__main__':
    N = 20
    hel = Helix.from_ndarray(onp.random.random((N, 5)))
    q = onp.random.choice([-1, 1], N)
    l = onp.random.random(N)
    B = 1. * onp.ones(N)

    f = position_from_helix

    df1 = jax.vmap(jax.jacfwd(f, argnums=0))
    df2 = jax.vmap(jax.jacrev(f, argnums=0))

    print(df1(hel, l, q, B).x.as_array.shape)
    print(df2(hel, l, q, B).x.as_array.shape)
