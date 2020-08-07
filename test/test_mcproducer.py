from helixgen.mcproducer import produce_and_serialize_to_json

import os

import jax.random as rjax

rng = rjax.PRNGKey(seed=0)


def produce(cart_cov_clu, cart_cov_hel):
    """ """
    decstr = 'D+ -> [K*(892)0 -> K- pi+] pi+'
    lbl='dkstpi'
    produce_and_serialize_to_json(
        rng, decstr=decstr, nevts=10, lbl=lbl,
        cart_cov_clu=cart_cov_clu, cart_cov_hel=cart_cov_hel)

    assert os.path.isfile(f'{lbl}.json')
    os.remove(f'{lbl}.json')

def test_produce_false_false():
    produce(False, False)

def test_produce_false_true():
    produce(False, True)

def test_produce_true_false():
    produce(True, False)

def test_produce_true_true():
    produce(True, True)
