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

from .phspdecay import generate


def produce_and_serialize_to_json(rng: jax.random.PRNGKey, decstr: str, nevts: int, lbl: str):
    """ """
    weights, genpcls = generate(rng, decstr, nevts)

    data = {'weights': weights.tolist()}

    for pname, info in genpcls.items():
        data[pname] = {
            'mom': info['mom'].as_array.tolist(),
            'pos': info['pos'].as_array.tolist(),
            'pdgid': info['pcl'].pdgid,
            'pdgname': info['pcl'].name,
            'pdgmass': info['pcl'].mass,
        }
        if 'clu' in info:  # photon
            data[pname].update({
                'clu': info['clu'].as_array.tolist(),
                'clucov': info['clucov'].tolist(),
                'meas_mom': info['meas_mom'].as_array.tolist(),
            })
        elif 'hel' in info:  # stable charged
            data[pname].update({
                'hel': info['hel'].as_array.tolist(),
                'helcov': info['helcov'].tolist(),
                'meas_pos': info['meas_pos'].as_array.tolist(),
                'meas_mom': info['meas_mom'].as_array.tolist(),
            })

    with open(os.path.join('./', f'{lbl}.json'), 'w') as ofile:
        ofile.write(json.dumps(data))
