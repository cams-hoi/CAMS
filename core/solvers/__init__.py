from .solver import Solver

def solver_entry(C):
    if C.config.get('solver', None):
        return globals()[C.config['solver']['type']](C)
    else:
        return Solver(C)
