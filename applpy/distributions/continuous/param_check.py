"""Parameter validation helpers for continuous distributions."""

from sympy import Symbol


def param_check(param):
    flag = True
    for element in param:
        if isinstance(element, Symbol):
            flag = False
    return flag


__all__ = ["param_check"]
