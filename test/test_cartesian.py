""" Unit tests for the cartesian package """

import pytest
from helixgen.cartesian import Position, Momentum
import jax.random as rjax
import jax.numpy as np

rng = rjax.PRNGKey(seed=0)

def test_position_ctor():
    """ """
    N = 100
    pos = Position.from_ndarray(rjax.uniform(rng, (N, 3)))
    
    assert pos.as_array.shape == (N, 3)


def test_momentum_ctor():
    """ """
    N = 100
    mom = Momentum.from_ndarray(rjax.uniform(rng, (N, 3)))
    
    assert mom.as_array.shape == (N, 3)
