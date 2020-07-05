"""
TODO: implement
https://github.com/zfit/phasespace/blob/master/phasespace/phasespace.py
with jax in order to get rid of tensorflow
"""

"""
Implementation of the Raubold and Lynch method to generate n-body events.
The code is based on the GENBOD function (W515 from CERNLIB), documented in
    F. James, Monte Carlo Phase Space, CERN 68-15 (1968)
"""

from typing import Union, Dict, Tuple, Optional, Callable
import jax.numpy as np


def kallen(Msq, m1sq, m2sq):
    """ Kallen's triangle function """
    return np.clip((Msq - m1sq - m2sq)**2 - 4*m1sq*m2sq, 0, a_max=None)


def two_body_momentum(Msq, m1sq, m2sq):
    """ Calculate the PDK (2-body phase space) function.
    Based on Eq. (9.17) in CERN 68-15 (1968). """
    return 0.5 * np.sqrt(kallen(Msq, m1sq, m2sq) / Msq)


class GenParticle:
    """ Representation of a particle """

    def __init__(self, name: str, mass: Union[Callable, int, float]) -> (None):
        self.name = name
        self.children = []
        self._mass = mass
        self._generate_called = False  # not yet called, children can be set

    def __repr__(self):
        return "<jaxphsp.GenParticle: name='{}' mass={} children=[{}]>" \
            .format(self.name,
                    f"{self._mass:.2f}" if self.has_fixed_mass else "variable",
                    ', '.join(child.name for child in self.children))

    def _do_names_clash(self, particles):
        """ TODO: use dict here instead of list to speed up """
        def get_list_of_names(part):
            output = [part.name]
            for child in part.children:
                output.extend(get_list_of_names(child))
            return output

        names_to_check = [self.name]
        for part in particles:
            names_to_check.extend(get_list_of_names(part))
        # Find top
        dup_names = {name for name in names_to_check if names_to_check.count(name) > 1}
        if dup_names:
            return dup_names
        return None


    def get_mass(self, min_mass: float = None, max_mass: float = None,
                 n_events: float = None) -> (float):
        """ If the particle is resonant, the mass function will be called with the
        `min_mass`, `max_mass` and `n_events` parameters """
        if self.has_fixed_mass:
            mass = self._mass
        else:
            min_mass = min_mass * np.ones(n_events)
            max_mass = max_mass * np.ones(n_events)
            mass = self._mass(min_mass, max_mass, n_events)
        return mass


    @property
    def has_fixed_mass(self):
        """ bool: Is the mass a callable function? """
        return not callable(self._mass)


    def set_children(self, *children):
        """ Assign children """
        if self._generate_called:
            raise RuntimeError("Cannot set children after the first call to `generate`.")
        if self.children:
            raise ValueError("Children already set!")
        if len(children) < 2:
            raise ValueError(f"Have to set at least 2 children, not {len(children)} for a particle to decay")
        # Check name clashes
        name_clash = self._do_names_clash(children)
        if name_clash:
            raise KeyError("Particle name {} already used".format(name_clash))
        self.children = children
        return self


    @property
    def has_children(self):
        """ bool: Does the particle have children? """
        return bool(self.children)


    @property
    def has_grandchildren(self):
        """bool: Does the particle have grandchildren?"""
        return self.has_children and any([child.has_children for child in self.children])

