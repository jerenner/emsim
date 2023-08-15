from typing import List, Tuple

from emsim.dataclasses import Event, IncidencePoint, MultiscaleEvent, Pixel, Rectangle


def read_file(filename: str) -> List[Event]:
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
                event.pixelset.pixels.append(pixel)
    return events


def read_multiscale_data(
    highres_filename: str,
    lowres_filename: str,
    highres_shape: Tuple[int],
    lowres_shape: Tuple[int],
    mm_x_range: Tuple[float],
    mm_y_range: Tuple[float]
) -> List[MultiscaleEvent]:
    highres_data: List[Event] = read_file(highres_filename)
    lowres_data: List[Event] = read_file(lowres_filename)
    assert len(highres_data) == len(lowres_data)

    multiscale_events = []
    for high, low in zip(highres_data, lowres_data):
        assert high.incidence == low.incidence
        event = MultiscaleEvent(
            high.incidence,
            lowres_image_size=Rectangle(0, lowres_shape[0], 0, lowres_shape[1]),
            highres_image_size=Rectangle(0, highres_shape[0], 0, highres_shape[1]),
            size_mm=Rectangle(mm_x_range[0], mm_x_range[1], mm_y_range[0], mm_y_range[1]),
            lowres_pixelset=low.pixelset,
            highres_pixelset=high.pixelset
            )
        multiscale_events.append(event)
    return multiscale_events
