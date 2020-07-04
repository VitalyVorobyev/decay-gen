""" """

from phasespace import GenParticle
from params import *


def generate():
    """  """
    pion = GenParticle('pi+', PION_MASS)
    kaon = GenParticle('K+', KAON_MASS)
    kstar = GenParticle('K*', KSTZ_MASS).set_children(pion, kaon)
    gamma = GenParticle('gamma', 0)
    dz = GenParticle('D0', D0_MASS).set_children(kstar, gamma)

    weights, particles = dz.generate(n_events=1000)

    return (weights, particles)
