from dataclasses import dataclass, field
from typing import Tuple, Union, List
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class IncidencePoint:
    id: int
    x: float
    y: float
    z: float
    e0: float

    def __post_init__(self):
        self.id = int(self.id)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.e0 = float(self.e0)

@dataclass
class Pixel:
    x: int
    y: int
    ionization_electrons: int

    def __post_init__(self):
        self.x = int(self.x)
        self.y = int(self.y)
        self.ionization_electrons = int(self.ionization_electrons)

@dataclass
class Event:
    incidence: IncidencePoint
    pixels: list[Pixel] = field(default_factory=list)
    array: Union[np.ndarray, sparse.spmatrix] = None

    def make_array(self, shape):
        self.array = make_array(self, shape)


def make_array(event: Event, shape: Tuple[int]):
    x, y, e = [], [], []
    for p in event.pixels:
        x.append(p.x)
        y.append(p.y)
        e.append(p.ionization_electrons)
    array = sparse.coo_array(
        (np.array(e), (np.array(x), np.array(y))),
        shape=shape)
    return array.tocsr()


def iterate_events(filename: str) -> List[Event]:
    events = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if "EV" in line:
                _, electron_id, electron_x, electron_y, electron_z, electron_e0 = line.split(" ")
                event = Event(IncidencePoint(electron_id, electron_x, electron_y, electron_z, electron_e0))
                events.append(event)
            else:
                pixel_x, pixel_y, ion_elecs = line.split(" ")
                pixel = Pixel(pixel_x, pixel_y, ion_elecs)
                event.pixels.append(pixel)
    return events


def read_file(filename: str, shape: Tuple[int]) -> List[Event]:
    events = iterate_events(filename)
    for event in events:
        event.make_array(shape)
    return events


def combine_event_arrays(events: List[Event]):
    return sum([event.array for event in events])
