""" """

from phasespace import GenParticle
from particle import Particle
from helix import Momentum, Position, Helix, helix_to_cartesian, cartesian_to_helix
import jax.numpy as np
import jax


def pname(lbl: str, name: str) -> (GenParticle):
    """ Particle from name """
    return GenParticle(name, Particle.findall(lbl)[0].mass)


def generate(nevt: int) -> (np.ndarray, dict):
    """ Generates decay chain """
    pion = pname('pi+', 'pi+')
    kaon = pname('K-', 'K-')
    kstar = pname('K*(892)0', 'K*0').set_children(pion, kaon)
    gamma = pname('gamma', 'gamma')
    dz = pname('D0', 'D0').set_children(kstar, gamma)

    weights, particles = dz.generate(n_events=nevt)

    return (weights.numpy(), particles)


def genmom_to_helix(genmom: np.ndarray) -> (Helix, np.ndarray):
    """  """
    nevt = genmom.shape[0]
    mom_pip = Momentum.from_ndarray(genmom[:,:-1])
    pos_pip = Position.from_ndarray(np.zeros((nevt, 3)))

    rng = jax.random.PRNGKey(seed=0)
    q = jax.random.choice(rng, [-1, 1], (nevt,))

    (hel, l) = cartesian_to_helix(pos_pip, mom_pip, q, B=1.5)

    return (hel, l)
