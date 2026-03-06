"""Distribution modules grouped by support type."""

from . import continuous as _continuous
from . import discrete as _discrete
from .continuous import __all__ as _continuous_all
from .discrete import __all__ as _discrete_all

__all__ = _continuous_all + _discrete_all

for _name in __all__:
    if hasattr(_continuous, _name):
        globals()[_name] = getattr(_continuous, _name)
    else:
        globals()[_name] = getattr(_discrete, _name)
