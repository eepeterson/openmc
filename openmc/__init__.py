from openmc.arithmetic import *
from openmc.cell import *
from openmc.checkvalue import *
from openmc.mesh import *
from openmc.element import *
from openmc.geometry import *
from openmc.nuclide import *
from openmc.macroscopic import *
from openmc.material import *
from openmc.plots import *
from openmc.region import *
from openmc.volume import *
from openmc.weight_windows import *
from openmc.surface import *
from openmc.universe import *
from openmc.source import *
from openmc.settings import *
from openmc.lattice import *
from openmc.filter import *
from openmc.filter_expansion import *
from openmc.trigger import *
from openmc.tally_derivative import *
from openmc.tallies import *
from openmc.mgxs_library import *
from openmc.executor import *
from openmc.statepoint import *
from openmc.summary import *
from openmc.particle_restart import *
from openmc.mixin import *
from openmc.plotter import *
from openmc.search import *
from openmc.polynomial import *
from openmc.tracks import *
from . import examples
from .config import *

# Import a few names from the model module
from openmc.model import rectangular_prism, hexagonal_prism, Model

def _simple_cls_name(obj):
    name = obj.__class__.__name__
    func = lambda x, y: '.'.join([x, y])
    mods = list(itertools.accumulate(obj.__module__.split('.'), func=func))
    filt = [name in eval(mod).__dict__ for mod in mods]
    mod = next(itertools.compress(mods, filt))
    return f'{mod}.{name}'


__version__ = '0.13.4-dev'
