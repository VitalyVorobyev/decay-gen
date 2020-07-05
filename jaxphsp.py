"""
TODO: implement
https://github.com/zfit/phasespace/blob/master/phasespace/phasespace.py
with jax in order to get rid of tensorflow
"""

import jax.numpy as np

def kallen(Msq, m1sq, m2sq):
    """ Kallen's triangle function """
    return np.clip((Msq - m1sq - m2sq)**2 - 4*m1sq*m2sq, 0, a_max=None)


def two_body_momentum(Msq, m1sq, m2sq):
    """ Calculate the PDK (2-body phase space) function.
    Based on Eq. (9.17) in CERN 68-15 (1968). """
    return 0.5 * np.sqrt(kallen(Msq, m1sq, m2sq) / Msq)


class GenParticle:
    pass
