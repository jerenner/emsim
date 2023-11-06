from typing import List, Tuple, Dict

import pandas as pd

from emsim.dataclasses import Event, IncidencePoint, IonizationElectronPixel, EnergyLossPixel
from emsim.geant.dataclasses import (GeantElectron, GeantGridsize, Trajectory,
                                     TrajectoryPoint)

def read_files(
    pixels_file: str, undiffused_pixels_file: str
) -> List[GeantElectron]:
    grid, pixel_events = read_pixelized_geant_output(pixels_file)
    undiffused_pixel_events = read_true_pixel_file(undiffused_pixels_file)

    electrons = []
    for pixel, undiffused in zip(
        pixel_events, undiffused_pixel_events
    ):
        assert pixel.incidence == undiffused.incidence
        assert pixel.incidence.id == undiffused.incidence.id
        elec = GeantElectron(
            id=pixel.incidence.id,
            incidence=pixel.incidence,
            pixels=pixel.pixelset,
            undiffused_pixels=undiffused.pixelset,
            grid=grid,
        )
        electrons.append(elec)

    return electrons


def read_pixelized_geant_output(filename: str) -> Tuple[GeantGridsize, List[Event]]:
    events = []
    gridsize = None
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if "#" in line:
                assert gridsize is None
                _, xmax_pixel, ymax_pixel, xmax_um, ymax_um = line.split(" ")
                xmax_pixel, ymax_pixel = int(xmax_pixel), int(ymax_pixel)
                xmax_um, ymax_um = float(xmax_um), float(ymax_um)
                gridsize = GeantGridsize(
                    xmax_pixel=xmax_pixel,
                    ymax_pixel=ymax_pixel,
                    xmax_um=xmax_um,
                    ymax_um=ymax_um,
                )
            elif "EV" in line:
                (
                    _,
                    electron_id,
                    electron_x,
                    electron_y,
                    electron_z,
                    electron_e0,
                ) = line.split(" ")
                event = Event(
                    IncidencePoint(
                        electron_id, electron_x, electron_y, electron_z, electron_e0
                    )
                )
                events.append(event)
            else:
                pixel_x, pixel_y, ion_elecs = line.split(" ")
                pixel = IonizationElectronPixel(pixel_x, pixel_y, ion_elecs)
                event.pixelset.append(pixel)
    return gridsize, events


def read_true_pixel_file(filename: str) -> List[Event]:
    with open(filename) as f:
        events = []
        for line in f:
            line = line.rstrip()
            if "#" in line:
                continue
            elif "EV" in line:
                (
                    _,
                    electron_id,
                    electron_x,
                    electron_y,
                    electron_z,
                    electron_e0,
                ) = line.split(" ")
                event = Event(
                    IncidencePoint(
                        electron_id, electron_x, electron_y, electron_z, electron_e0
                    )
                )
                events.append(event)
            else:
                pixel_x, pixel_y, ion_elecs = line.split(" ")
                pixel = EnergyLossPixel(pixel_x, pixel_y, ion_elecs)
                event.pixelset.append(pixel)
    return events


def read_trajectory_file(filename: str) -> List[Trajectory]:
    with open(filename) as f:
        data = []
        for line in f:
            line = line.rstrip()
            if "new_e-" in line:
                electron_id = len(data)
                t = 0
                _, x0, y0, z0, pz0, e0 = line.split(" ")
                traj = Trajectory(
                    electron_id=electron_id,
                    x0=x0,
                    y0=y0,
                    z0=z0,
                    pz0=pz0,
                    e0=e0,
                )
                data.append(traj)
                traj.append(
                    TrajectoryPoint(
                        electron_id=electron_id,
                        t=t,
                        x=x0,
                        y=y0,
                        z=z0,
                        edep=0,
                        x0=x0,
                        y0=y0,
                        z0=z0,
                        e0=e0,
                    )
                )
            else:
                x, y, z, edep, electron_id, x0, y0 = line.split(" ")
                assert int(electron_id) == traj[0].electron_id
                assert float(x0) == float(traj[0].x0)
                assert float(y0) == float(traj[0].y0)
                t = len(traj)
                traj.append(
                    TrajectoryPoint(
                        electron_id=electron_id,
                        t=t,
                        x=x,
                        y=y,
                        z=z,
                        edep=edep,
                        x0=x0,
                        y0=y0,
                        z0=z0,
                        e0=e0,
                    )
                )
    return data


def trajectories_to_df(data: List[Trajectory]) -> pd.DataFrame:
    df = pd.DataFrame([pt for traj in data for pt in traj])
    df = df.set_index(["electron_id", "t"])
    return df