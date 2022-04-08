"""Package root.
"""

import typing


class Rectangle(typing.NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Point(typing.NamedTuple):
    x: float
    y: float
