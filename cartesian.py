""" Tools for track helix manipulations

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

from typing import NamedTuple

import jax.numpy as np
import jax

rng = jax.random.PRNGKey(seed=0)
    
dtype = np.ndarray

speedOfLight = 29.9792458  # [cm / ns]

def alpha(B):
    """ Magnetic field to smth """
    return 1.0 / (B * speedOfLight) * 1e4


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
    def r(self) -> (dtype):
        """ Total length """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


    @property
    def costh(self) -> (dtype):
        """ Cosine of the polar angle """
        return self.z / self.r

    @property
    def phi(self) -> (dtype):
        """ Azimuth angle. arctan2 chooses the quadrant correctly """
        return np.arctan2(self.y, self.x)

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

    @property
    def ptot(self) -> (dtype):
        """ Total momentum """
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def costh(self) -> (dtype):
        """ Cosine of the polar angle """
        return self.pz / self.ptot

    @property
    def phi(self) -> (dtype):
        """ Azimuth angle. arctan2 chooses the quadrant correctly """
        return np.arctan2(self.py, self.px)


    def px0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ Helper function for helix """
        return self.px + pos.y * q * alpha(B)


    def py0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ Helper function for helix """
        return self.py - pos.x * q * alpha(B)


    def pt0(self, pos: Position, q: dtype, B: float)\
        -> (dtype):
        """ Helper function for helix """
        return np.sqrt(self.px0(pos, q, B)**2 +
                       self.py0(pos, q, B)**2)


    def phi0pt0(self, pos: Position, q: dtype, B: float) -> (dtype):
        """ Helper function that calculates phi0 and pt0 efficiently """
        px0_ = self.px0(pos, q, B)
        py0_ = self.py0(pos, q, B)
        return (np.arctan2(py0_, px0_), np.sqrt(px0_**2 + py0_**2))
