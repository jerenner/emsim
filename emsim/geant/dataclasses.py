from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from emsim.dataclasses import (
    IonizationElectronPixel,
    EnergyLossPixel,
    IncidencePoint,
    PixelSet,
)


@dataclass
class GeantGridsize:
    xmax_pixel: int
    ymax_pixel: int
    xmax_um: float
    ymax_um: float
    xmin_pixel: Optional[int] = 0
    ymin_pixel: Optional[int] = 0
    xmin_um: Optional[float] = 0.0
    ymin_um: Optional[float] = 0.0

    @property
    def pixel_size_um(self):
        x_size = (self.xmax_um - self.xmin_um) / (self.xmax_pixel - self.xmin_pixel)
        y_size = (self.ymax_um - self.ymin_um) / (self.ymax_pixel - self.ymin_pixel)
        return x_size, y_size


@dataclass
class TrajectoryPoint:
    electron_id: int
    t: int
    x: float
    y: float
    z: float
    edep: float
    x0: Optional[float] = None
    y0: Optional[float] = None
    z0: Optional[float] = None
    e0: Optional[float] = None

    def __post_init__(self):
        self.electron_id = int(self.electron_id)
        self.t = int(self.t)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.edep = float(self.edep)
        self.x0 = float(self.x0) if self.x0 else None
        self.y0 = float(self.y0) if self.y0 else None
        self.z0 = float(self.z0) if self.z0 else None
        self.e0 = float(self.e0) if self.e0 else None


@dataclass
class Trajectory:
    electron_id: int
    x0: float
    y0: float
    z0: float
    pz0: float
    e0: float
    _points: List[TrajectoryPoint] = field(default_factory=list)

    def __post_init__(self):
        self.x0 = float(self.x0)
        self.y0 = float(self.y0)
        self.z0 = float(self.z0)
        self.pz0 = float(self.pz0)
        self.e0 = float(self.e0)

    def __getitem__(self, item):
        return self._points[item]

    def append(self, item):
        self._points.append(item)

    def __len__(self):
        return len(self._points)

    def as_array(self):
        return np.stack(
            [
                np.array([point.x, point.y, point.z, point.edep], dtype=np.float32)
                for point in self._points
            ],
            0,
        )

    def localize(self, x_offset: float, y_offset: float):
        new_points = [
            replace(
                point,
                x=point.x - x_offset,
                y=point.y - y_offset,
                x0=point.x0 - x_offset if point.x0 else None,
                y0=point.y0 - y_offset if point.y0 else None,
            )
            for point in self._points
        ]
        return replace(
            self, x0=self.x0 - x_offset, y0=self.y0 - y_offset, _points=new_points
        )


@dataclass
class GeantElectron:
    id: int
    incidence: IncidencePoint
    trajectory: Trajectory
    pixels: PixelSet
    undiffused_pixels: PixelSet
    grid: GeantGridsize