from emsim.dataclasses import IncidencePoint, PixelSet, Rectangle


from dataclasses import dataclass, field
from functools import cached_property
from typing import Union


@dataclass
class MultiscaleFrame:
    mm: Rectangle
    lowres: Rectangle
    highres: Rectangle

    @cached_property
    def highres_pixel_size_mm(self):
        return (
            self.mm.width() / self.highres.width(),
            self.mm.height() / self.highres.height()
        )

    @cached_property
    def lowres_pixel_size_mm(self):
        return (
            self.mm.width() / self.lowres.width(),
            self.mm.height() / self.lowres.height()
        )

    def lowres_to_highres_scaling(self):
        low = self.lowres_pixel_size_mm
        high = self.highres_pixel_size_mm
        return (low[0] / high[0], low[1] / high[1])

    def mm_to_highres(self, x_mm: float, y_mm: float):
        x = (x_mm - self.mm.xmin) / self.highres_pixel_size_mm[0]
        y = (y_mm - self.mm.ymin) / self.highres_pixel_size_mm[1]
        return x, y

    def mm_to_lowres(self, x_mm: float, y_mm: float):
        x = (x_mm - self.mm.xmin) / self.lowres_pixel_size_mm[0]
        y = (y_mm - self.mm.ymin) / self.lowres_pixel_size_mm[1]
        return x, y

    def lowres_coord_to_mm(
        self, x_lowres: Union[int, float], y_lowres: Union[int, float],
    ):
        x = x_lowres * self.lowres_pixel_size_mm + self.mm.xmin
        y = y_lowres * self.lowres_pixel_size_mm + self.mm.ymin
        return x, y

    def lowres_index_to_mm(
        self, x_lowres: int, y_lowres: int
    ):
        if not isinstance(x_lowres, int) or not isinstance(y_lowres, int):
            raise ValueError(
                f"Got a non-int value for a pixel index: {(x_lowres, y_lowres)=}")
        x = x_lowres + 0.5
        y = y_lowres + 0.5
        return self.lowres_coord_to_mm(x, y)

    def highres_coord_to_mm(
        self, x_highres: Union[int, float], y_highres: Union[int, float],
    ):
        x = x_highres * self.highres_pixel_size_mm + self.mm.xmin
        y = y_highres * self.highres_pixel_size_mm + self.mm.ymin
        return x, y

    def highres_index_to_mm(
        self, x_highres: int, y_highres: int
    ):
        if not isinstance(x_highres, int) or not isinstance(y_highres, int):
            raise ValueError(
                f"Got a non-int value for a pixel index: {(x_highres, y_highres)=}")
        x = x_highres + 0.5
        y = y_highres + 0.5
        return self.highres_coord_to_mm(x, y)


@dataclass
class MultiscalePixelSet:
    incidence: IncidencePoint
    lowres_image_size: Rectangle
    highres_image_size: Rectangle
    size_mm: Rectangle
    lowres_pixelset: PixelSet = field(default_factory=PixelSet)
    highres_pixelset: PixelSet = field(default_factory=PixelSet)

    @property
    def id(self):
        return self.incidence.id
