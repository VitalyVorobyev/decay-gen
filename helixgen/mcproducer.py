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
from .resolution import apply_resolution
from .resolution import apply_resolution_neutral, apply_resolution_neutral_cartesian
from .resolution import apply_resolution_charged, apply_resolution_charged_cartesian

def produce_and_serialize_to_json(
        rng: jax.random.PRNGKey,
        decstr: str,
        nevts: int,
        lbl: str,
        cart_cov_clu=False,
        cart_cov_hel=False):
    """ """
    smerer = lambda rng, gp: apply_resolution(
        rng, gp,
        (apply_resolution_neutral_cartesian if cart_cov_clu else apply_resolution_charged),
        (apply_resolution_charged_cartesian if cart_cov_hel else apply_resolution_charged)
    )
    weights, genpcls = generate(rng, decstr, nevts, smerer)

    data = {'weights': weights.tolist()}

    for pname, info in genpcls.items():
        data[pname] = {
            key: (val.tolist() if hasattr(val, 'as_array') else\
                  val.as_array.tolist()) for key, val in info}

    with open(os.path.join('./', f'{lbl}.json'), 'w') as ofile:
        ofile.write(json.dumps(data))
