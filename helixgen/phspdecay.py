""" Tools for decay kinematics generation

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: August 2020
"""

from typing import Callable

import tensorflow as tf
import jax.numpy as np
import jax.random as rjax

from phasespace import GenParticle
from particle import Particle

from .cartesian import Momentum, Position


def pname(lbl: str, name: str) -> (GenParticle, Particle):
    """ GenParticle and Particle from particle label """
    p = Particle.findall(lbl)[0]
    return (GenParticle(name, p.mass), p)


def dp_to_kstz_pip(nevt: int) -> ((tf.Tensor, tf.Tensor), dict):
    """ D+ -> [K*0 -> K- pi+] pi+ decay generator """
    pion, ppion = pname('pi+', 'pi+')
    kaon, pkaon = pname('K-', 'K-')
    kstar, pkstar = pname('K*(892)0', 'K*0')
    kstar.set_children(pion, kaon)
    pion_dplus, dpion_dplus = pname('pi+', 'D+_pi+')
    dplus, _ = pname('D+', 'D+')
    dplus.set_children(kstar, pion_dplus)

    genpcls = {
        'pi+': {'pcl': ppion},
        'K-': {'pcl': pkaon},
        'K*0': {'pcl': pkstar},
        'D+_pi+': {'pcl': dpion_dplus},
        'root': {'gpcl': dplus},
    }

    return (dplus.generate(n_events=nevt), genpcls)


def dz_to_ks_pip_pim(nevt: int) -> ((tf.Tensor, tf.Tensor), dict):
    """ D0 -> [Ks0 -> pi+ pi-] pi+ pi- decay generator """
    pip_ks, ppip_ks = pname('pi+', 'pi+_Ks0')
    pim_ks, ppim_ks = pname('pi-', 'pi-_Ks0')
    ks, pks = pname('K(S)0', 'Ks0')
    ks.set_children(pip_ks, pim_ks)
    pip, ppip = pname('pi+', 'pi+')
    pim, ppim = pname('pi-', 'pi-')
    dz, _ = pname('D0', 'D0')
    dz.set_children(ks, pip, pim)

    genpcls = {
        'pi+_Ks0': {'pcl': ppip_ks},
        'pi-_Ks0': {'pcl': ppim_ks},
        'Ks0': {'pcl': pks},
        'pi+': {'pcl': ppip},
        'pi-': {'pcl': ppim},
        'root': {'gpcl': dz},
    }

    return (dz.generate(n_events=nevt), genpcls)


def generate_phsp(decstr: str, nevt: int) -> (np.ndarray, dict):
    """ TODO: temporary generator that produces fixed decay chain.
    Should be able to parse decay string and produce decay chain accordingly
    """
    # (weights, particles), genpcls = dp_to_kstz_pip(nevt)
    (weights, particles), genpcls = dz_to_ks_pip_pim(nevt)

    for key, mom in particles.items():
        genpcls[key]['mom'] = Momentum.from_ndarray(mom.numpy()[:, :-1])

    return (np.array(weights.numpy()), genpcls)


def generate_positions(rng, genpcls, pos0, pcl=None):
    """ Generates position according to the momentum direction and lifetime
        Traverses decay tree recursively """
    if pcl is None:
        genpcls['root']['pos'] = Position.from_ndarray(pos0)
        pcl = genpcls['root']['gpcl']
    else:
        genpcls[pcl.name]['pos'] = Position.from_ndarray(pos0)

    for ch in pcl.children:
        particle = genpcls[ch.name]['pcl']
        if particle.lifetime > 0.0001 and particle.lifetime < 1:
            mom = genpcls[ch.name]['mom']
            nevt = mom.size
            rng, key = rjax.split(rng)
            time = particle.lifetime * rjax.exponential(key, (nevt, 1))
            # TODO: add gamma factor multiplier here (relativistic correction)
            pos0 = pos0 + mom.velocity(particle.mass) * time
        generate_positions(rng, genpcls, pos0, ch)


def generate(rng: rjax.PRNGKey, decstr: str, nevt: int, smearer: Callable):
    """ Runs all steps of event generation:
     1. phase space momentum generator
     2. positions generator
     3. helix smeared by resolution for charged particles
     4. clusters smeared by resolution for photons
     5. ("measured") position and momentum from smeared helixes and clusters
     6. covariance matrix for each helix and cluster
     All the reults are saved in dictionary
    """
    ws, genpcls = generate_phsp(decstr, nevt)
    # root particle position
    pos0 = np.zeros((nevt, 3))
    generate_positions(rng, genpcls, pos0)
    genpcls.pop('root')
    return (ws, smearer(rng, genpcls))
