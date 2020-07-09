from helixgen.mcproducer import produce_and_serialize_to_json

import json

def test_produce_and_serialize_to_json():
    """ """
    produce_and_serialize_to_json('D+ -> [K*(892)0 -> K- pi+] pi+', 10, 'dkstpi')

