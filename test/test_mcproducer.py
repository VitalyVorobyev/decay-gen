from helixgen.mcproducer import produce_and_serialize_to_json

import jax.random as rjax
# import json

rng = rjax.PRNGKey(seed=0)

def test_produce_and_serialize_to_json():
    """ """
    produce_and_serialize_to_json(rng, 'D+ -> [K*(892)0 -> K- pi+] pi+', 10, 'dkstpi')

