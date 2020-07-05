#! /usr/bin/env python

""" Main tools for the MC event production

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

import os
import json
from particle import Particle

from phspdecay import generate, genmom_to_helix

stable_charged = set(['pi+', 'pi-', 'e+', 'e-', 'p', 'anti-p', 'K+', 'K-', 'mu+', 'mu-'])
stable_neutral = set(['gamma', 'n', 'anti-n'])


def produce(decstr: str, nevts: int, lbl: str):
    """ """
    weights, particles, namedict = generate(decstr, nevts)

    data = {'weights': weights.tolist()}

    for pname, pmom in particles.items():
        pmom = pmom.numpy()
        pdgname = namedict[pname]
        data[pname] = {'momgen': pmom.tolist()}
        if pdgname in stable_charged:
            pos, hel, l = genmom_to_helix(pmom)
            data[pname].update({
                'posgen': pos.as_array.tolist(),
                'helix': hel.as_array.tolist(),
                'l': l.tolist(),
                'pdgtype': pdgname,
                'pdgid': Particle.findall(pdgname)[0].pdgid,
            })

    with open(os.path.join('data', f'{lbl}.json'), 'w') as ofile:
        ofile.write(json.dumps(data))


if __name__ == '__main__':
    produce('D+ -> [K*(892)0 -> K- pi+] pi+', 100, 'dkstpi')
