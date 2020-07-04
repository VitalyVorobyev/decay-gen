""" """

from collections import namedtuple

from particle import Particle
from particle.pdgid import literals as lid

from phasespace import GenParticle

Pcl = namedtuple('Pcl', ['name', 'pdgid'])
Decay = namedtuple('Decay', ['moth', 'daug'])

decays = {
    'pi0 -> gamma gamma' : Decay(
        Pcl('pi0', lid.pi_0),
        [
            Pcl('gam1', lid.gamma),
            Pcl('gam2', lid.gamma),
        ]
    ),

    'Ks0 -> pi+ pi-' : Decay(
        Pcl('Ks0', lid.K_S_0),
        [
            Pcl('pi+', lid.pi_plus),
            Pcl('pi-', lid.pi_minus),
        ]
    ),

    'D0 -> Ks0 pi+ pi-': Decay(
        Pcl('D0', lid.D_0),
        [
            'Ks0 -> pi+ pi-',
            Pcl('pi+', lid.pi_plus),
            Pcl('pi-', lid.pi_minus),
        ]
    )
}

def parse_decay(key):
    """ Construct GenParticle from decay structure """
    dec = decays['key']
    # TODO
