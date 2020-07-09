""" """

from phasespace import GenParticle
from particle import Particle

from .cartesian import Momentum, Position
from .helix import Helix, cartesian_to_helix, helix_to_cartesian, sample_helix_resolution
from .cluster import Cluster, cartesian_to_cluster, sample_cluster_resolution, momentum_from_cluster

import jax.numpy as np
import jax.random as rjax
import jax


def pname(lbl: str, name: str) -> (GenParticle, Particle):
    """ GenParticle and Particle from particle label """
    p = Particle.findall(lbl)[0]
    return (GenParticle(name, p.mass), p)


def generate_phsp(decstr: str, nevt: int) -> (np.ndarray, dict):
    """ TODO: temporary generator that produces fixed decay chain.
    Should be able to parse decay string and produce decay chain accordingly """
    pion, ppion = pname('pi+', 'pi+')
    kaon, pkaon = pname('K-', 'K-')
    kstar, pkstar = pname('K*(892)0', 'K*0')
    kstar.set_children(pion, kaon)
    pion_dplus, dpion_dplus = pname('pi+', 'D0_pi+')
    dplus, _ = pname('D+', 'D+')
    dplus.set_children(kstar, pion_dplus)

    weights, particles = dplus.generate(n_events=nevt)

    genpcls = {
        'pi+': {'pcl': ppion},
        'K-': {'pcl': pkaon},
        'K*0': {'pcl': pkstar},
        'D0_pi+': {'pcl': dpion_dplus},
        'root': {'gpcl': dplus},
    }

    for key, mom in particles.items():
        genpcls[key]['mom'] = Momentum.from_ndarray(mom.numpy()[:,:-1])

    return (np.array(weights.numpy()), genpcls)


@jax.vmap
def direction(mom):
    """ """
    return mom / np.sqrt(np.sum(mom**2))


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
            nevt = mom.shape[0]
            rng, key = rjax.split(rng)
            time = particle.lifetime*rjax.exponential(key, (nevt,))
            pos0 = pos0 + direction(mom) * time
        generate_positions(rng, genpcls, pos0, ch)


def apply_resolution_charged(rng: jax.random.PRNGKey, pos: Position, mom: Momentum, pcl: Particle)\
    -> (Helix, np.ndarray, Position, Momentum):
    """ returns measured helix, helix covariance, position and momentum """
    q, B = pcl.charge, 1.5
    hel, _ = cartesian_to_helix(pos, mom, q=q, B=B)
    hel, cov = sample_helix_resolution(rng, hel)
    pos, mom = helix_to_cartesian(hel, l=0., q=q, B=B)

    return (hel, cov, pos, mom)


def apply_resolution_photon(rng: jax.random.PRNGKey, pos: Position, mom: Momentum)\
    -> (Cluster, np.ndarray, Momentum):
    """ returns measured cluster, cluster covariance, and momentum """
    clu = cartesian_to_cluster(pos, mom)
    clu, cov = sample_cluster_resolution(rng, clu)
    mom = momentum_from_cluster(clu)

    return (clu, cov, mom)


def generate(rng: jax.random.PRNGKey, decstr: str, nevt: int):
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

    stable_charged = set(['pi+', 'pi-', 'e+', 'e-', 'p', 'anti-p', 'K+', 'K-', 'mu+', 'mu-'])

    for name, data in genpcls.items():
        if data['pcl'].name == 'gamma':
            clu, cov, mom = apply_resolution_photon(rng, data['pos'], data['mom'])
            genpcls[name].update({
                'clu': clu,
                'clucov': cov,
                'meas_mom': mom
            })
        elif data['pcl'].name in stable_charged:
            hel, cov, pos, mom = apply_resolution_charged(rng, data['pos'], data['mom'], data['pcl'])
            genpcls[name].update({
                'hel': hel,
                'helcov': cov,
                'meas_pos': pos,
                'meas_mom': mom,
            })

    return (ws, genpcls)
