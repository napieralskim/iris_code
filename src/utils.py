from typing import NamedTuple


class Coords(NamedTuple):
    x: int
    y: int

    @classmethod
    def from_floats(cls, coords: CoordsF):
        return cls(x=int(coords.x), y=int(coords.y))


class CoordsF(NamedTuple):
    x: float
    y: float

    @classmethod
    def from_ints(cls, coords: Coords):
        return cls(x=coords.x, y=coords.y)
