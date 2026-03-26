"""Integration scheme registry for depletion.

The actual scheme logic (transport, matrix exponential, iteration) is
implemented in C++ (see ``src/depletion_scheme.cpp``).  This module
provides only the set of valid scheme names so that the Python
:class:`~openmc.deplete.DepletionManager` can validate user input
without duplicating the algorithmic definitions.
"""

__all__ = ['SCHEMES', 'FALLBACK_SCHEMES']

#: Valid integration scheme names recognised by the C++ depletion kernel.
SCHEMES: frozenset[str] = frozenset({
    'predictor',
    'cecm',
    'celi',
    'cf4',
    'epc_rk4',
    'leqi',
    'si_celi',
    'si_leqi',
})

#: Schemes that require a lower-order fallback on the first timestep
#: (because they reference the previous step's BOS matrix, which does not
#: exist at step 0).
FALLBACK_SCHEMES: dict[str, str] = {
    'leqi': 'celi',
    'si_leqi': 'si_celi',
}
