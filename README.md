![.github/workflows/test_on_push.yaml](https://github.com/VitalyVorobyev/decay-gen/workflows/.github/workflows/test_on_push.yaml/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/VitalyVorobyev/decay-gen/branch/master/graph/badge.svg)](https://codecov.io/gh/VitalyVorobyev/decay-gen)

# HelixGen

Toy Monte-Carlo generator of particle decays featuring simulation of track helix and photon cluster. Experimental resolution is applied at the level of helixes and clusters. HelixGen may be useful for testing kinematic fitting algorithms.

Phase space decay generation is done with the [zfit/phasespace](https://github.com/zfit/phasespace) package and Tensorflow backend. Flight lengths and resolution sampling is done with [Jax](https://github.com/google/jax).

HelixGen can serialize event generated to json. Example of events generation can be found in [notebook](exmaples/genexample.ipynb).
