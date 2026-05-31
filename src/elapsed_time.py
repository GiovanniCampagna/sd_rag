from time import time
from typing import Callable, TypeVar, ParamSpec, Concatenate

T = TypeVar('T')
P = ParamSpec('P')

def timeit(
    f: Callable[Concatenate[P], T], 
    *args: P.args, 
    **kwargs: P.kwargs,
) -> tuple[T, int]:

    t0 = int(1000 * time())
    v = f(*args, **kwargs)
    t1 = int(1000 * time())
    elapsed_ms = t1 - t0
    return v, elapsed_ms

def start_timer():

    t0 = int(1000 * time())

    def stop() -> int:
        t1 = int(1000 * time())
        elapsed_ms = t1 - t0
        return elapsed_ms

    return stop
