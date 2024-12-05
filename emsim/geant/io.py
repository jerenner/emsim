from typing import List, Tuple

import numpy as np
import h5py
import pandas as pd

from emsim.dataclasses import (
    Event,
    IncidencePoint,
    IonizationElectronPixel,
    EnergyLossPixel,
    PixelSet,
)
from emsim.geant.dataclasses import (
    GeantElectron,
    GeantGridsize,
    Trajectory,
    TrajectoryPoint,
)


def read_files(pixels_file: str, trajectory_file: str = None) -> List[GeantElectron]:
    grid, pixel_events = read_pixelized_geant_output(pixels_file)
    if trajectory_file is not None:
        trajectories = read_trajectory_file(trajectory_file)
    else:
        trajectories = [None] * len(pixel_events)

    electrons = []
    for event, trajectory in zip(pixel_events, trajectories):
        if len(event.pixelset._pixels) > 0:
            elec = GeantElectron(
                id=event.incidence.id,
                incidence=event.incidence,
                pixels=event.pixelset,
                grid=grid,
                trajectory=trajectory,
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


def convert_electron_pixel_file_to_hdf5(pixels_file: str, h5_file: str, h5_mode="a"):
    electrons = read_files(pixels_file)
    if len(electrons) == 0:
        raise ValueError(f"Found no valid electrons in file {pixels_file}")
    with h5py.File(h5_file, h5_mode) as file:
        file.create_dataset("num_electrons", data=len(electrons), dtype=np.uint32)
        first = electrons[0]
        grid_group = file.create_group("grid")
        for k, v in first.grid.__dict__.items():
            grid_group.create_dataset(k, data=v, dtype=np.uint16)
        for i, electron in enumerate(electrons):
            group = file.create_group(str(i))
            group["id"] = i
            incidence_group = group.create_group("incidence")
            for k, v in electron.incidence.__dict__.items():
                incidence_group.create_dataset(k, data=v)
            # group.create_dataset("id", data=electron.id)
            # group.create_dataset("incidence/x", data=electron.incidence.x)
            # group.create_dataset("incidence/y", data=electron.incidence.y)
            # group.create_dataset("incidence/z", data=electron.incidence.z)
            # group.create_dataset("incidence/e0", data=electron.incidence.e0)

            pixel_group = group.create_group("pixels")
            pixels: List[IonizationElectronPixel] = electron.pixels._pixels
            pixel_group.create_dataset("x", data=np.array([p.x for p in pixels]))
            pixel_group.create_dataset("y", data=np.array([p.y for p in pixels]))
            pixel_group.create_dataset(
                "ionization_electrons", data=np.array([p.data for p in pixels])
            )


def read_electrons_from_hdf(
    h5_file: str, electron_ids: list[int]
) -> list[GeantElectron]:
    with h5py.File(h5_file, "r") as f:
        electrons = [read_single_electron_from_hdf(f, id) for id in electron_ids]
    return electrons


def read_single_electron_from_hdf(
    h5_fileptr: h5py.File, electron_id: int
) -> GeantElectron:
    electron_group: h5py.Group = h5_fileptr[str(electron_id)]
    assert electron_id == electron_group["id"][()]
    incidence_group: h5py.Group = electron_group["incidence"]
    incidence_point = IncidencePoint(
        **{key: incidence_group[key][()] for key in incidence_group.keys()}
    )

    grid_group = h5_fileptr["grid"]
    grid = GeantGridsize(**{key: grid_group[key][()] for key in grid_group.keys()})

    pixel_group: h5py.Group = electron_group["pixels"]
    pixel_x = pixel_group["x"][:]
    pixel_y = pixel_group["y"][:]
    pixel_ionization_electrons = pixel_group["ionization_electrons"][:]

    pixels = [
        IonizationElectronPixel(x, y, elecs)
        for x, y, elecs in zip(
            pixel_x, pixel_y, pixel_ionization_electrons, strict=True
        )
    ]

    return GeantElectron(
        id=electron_id,
        incidence=incidence_point,
        pixels=PixelSet(pixels),
        grid=grid,
        trajectory=None,
    )


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
