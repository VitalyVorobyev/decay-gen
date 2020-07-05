""" """

from phasespace import GenParticle
from particle import Particle
from helix import Momentum, Position, Helix, helix_to_cartesian, cartesian_to_helix
import jax.numpy as np
import jax


def pname(lbl: str, name: str) -> (GenParticle):
    """ Particle from name """
    return GenParticle(name, Particle.findall(lbl)[0].mass)


def generate(decstr: str, nevt: int) -> (np.ndarray, dict):
    """ Generates decay chain
    FIXME: implement decay generation for a given decay string
    """
    pion = pname('pi+', 'pi+')
    kaon = pname('K-', 'K-')
    kstar = pname('K*(892)0', 'K*0').set_children(pion, kaon)
    pion_dplus = pname('pi+', 'D0_pi+')
    dplus = pname('D+', 'D+').set_children(kstar, pion_dplus)
    # dz = pname('D0', 'D0').set_children(kstar, gamma)
    # gamma = pname('gamma', 'gamma')
    # dz = pname('D0', 'D0').set_children(kstar, gamma)

    namedict = {
        'pi+': 'pi+',
        'K-': 'K-',
        'K*0': 'K*(892)0',
        'D0_pi+': 'pi+',
    }

    weights, particles = dplus.generate(n_events=nevt)

    return (weights.numpy(), particles, namedict)


def genmom_to_helix(genmom: np.ndarray) -> (Helix, np.ndarray):
    """  """
    nevt = genmom.shape[0]
    mom = Momentum.from_ndarray(genmom[:,:-1])
    pos = Position.from_ndarray(np.zeros((nevt, 3)))

    rng = jax.random.PRNGKey(seed=0)
    q = jax.random.choice(rng, [-1, 1], (nevt,))

    hel, l = cartesian_to_helix(pos, mom, q, B=1.5)

    return (pos, hel, l)
