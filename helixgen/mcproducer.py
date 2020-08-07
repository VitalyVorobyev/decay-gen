#! /usr/bin/env python

""" Main tools for the MC event production

    Contributors:
        - Vitaly Vorobyev (vit.vorobiev@gmail.com)

    Created: July 2020
    Modified: July 2020
"""

import os
import json
import jax

from particle import Particle

from .phspdecay import generate
from . import resolution as res

def produce_and_serialize_to_json(
        rng: jax.random.PRNGKey,
        decstr: str, nevts: int, lbl: str,
        cart_cov_clu=False, cart_cov_hel=False):
    """ Main routine for event generation """
    smearer = lambda rng, gp: res.apply_resolution(
        rng, gp,
        (res.apply_resolution_neutral_cartesian if cart_cov_clu else res.apply_resolution_charged),
        (res.apply_resolution_charged_cartesian if cart_cov_hel else res.apply_resolution_charged)
    )
    weights, genpcls = generate(rng, decstr, nevts, smearer)

    data = {'weights': weights.tolist()}

    for pname, info in genpcls.items():
        data[pname] = {
            'pdgid': info['pcl'].pdgid,
            'pdgname': info['pcl'].name,
            'pdgmass': info['pcl'].mass,
        }
        for key, val in info.items():
            if isinstance(val, Particle):
                continue
            if hasattr(val, 'as_array'):
                data[pname][key] = val.as_array.tolist()
            else:
                data[pname][key] = val.tolist()

    with open(os.path.join('./', f'{lbl}.json'), 'w') as ofile:
        ofile.write(json.dumps(data))
