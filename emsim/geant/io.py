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
        file.create_dataset(
            "id",
            data=np.array([electron.id for electron in electrons], dtype=np.uint32),
        )

        # grid data
        grid_group = file.create_group("grid")
        first = electrons[0]
        for k, v in first.grid.__dict__.items():
            grid_group.create_dataset(k, data=v)

        # incidence point data
        incidence = {
            k: np.array([elec.incidence.__dict__[k] for elec in electrons])
            for k in electrons[0].incidence.__dict__
        }
        incidence_group = file.create_group("incidence")
        incidence_group.create_dataset("id", dtype=np.uint32, data=incidence["id"])
        incidence_group.create_dataset("x", dtype=float, data=incidence["x"])
        incidence_group.create_dataset("y", dtype=float, data=incidence["y"])
        incidence_group.create_dataset("z", dtype=float, data=incidence["z"])
        incidence_group.create_dataset("e0", dtype=float, data=incidence["e0"])

        # pixels data
        pixels_group = file.create_group("pixels")

        def _pixels_data(electrons: list[GeantElectron], field: str, dtype=np.dtype):
            return [
                np.array([pixel.__dict__[field] for pixel in elec.pixels], dtype=dtype)
                for elec in electrons
            ]

        pixels_group.create_dataset(
            "x",
            dtype=h5py.vlen_dtype(np.uint16),
            data=_pixels_data(electrons, "x", np.uint16),
        )
        pixels_group.create_dataset(
            "y",
            dtype=h5py.vlen_dtype(np.uint16),
            data=_pixels_data(electrons, "y", np.uint16),
        )
        pixels_group.create_dataset(
            "ionization_electrons",
            dtype=h5py.vlen_dtype(np.uint32),
            data=_pixels_data(electrons, "ionization_electrons", np.uint32),
        )


def read_electrons_from_hdf(
    h5_file: str, electron_ids: np.ndarray[int]
) -> list[GeantElectron]:
    n_electrons = len(electron_ids)
    sort_order = np.argsort(electron_ids)
    unsort_order = np.argsort(sort_order)
    with h5py.File(h5_file, "r") as f:
        grid_group = f["grid"]
        grid = GeantGridsize(**{key: grid_group[key][()].item() for key in grid_group.keys()})

        incidence = {
            "id": np.zeros(n_electrons, dtype=np.uint32),
            "x": np.zeros(n_electrons, dtype=float),
            "y": np.zeros(n_electrons, dtype=float),
            "z": np.zeros(n_electrons, dtype=float),
            "e0": np.zeros(n_electrons, dtype=float),
        }
        for k, v in incidence.items():
            v[sort_order] = f["incidence"][k][electron_ids[sort_order]]

        pixels = {
            k: f["pixels"][k][electron_ids[sort_order]][unsort_order]
            for k in f["pixels"].keys()
        }

        electrons = []
    for i, electron_id in enumerate(electron_ids):
        incidence_i = IncidencePoint(**{key: incidence[key][i] for key in incidence})

        pixels_i = {key: pixels[key][i] for key in pixels}
        pixels_i = [
            IonizationElectronPixel(x_i, y_i, elecs_i)
            for x_i, y_i, elecs_i in zip(
                pixels_i["x"], pixels_i["y"], pixels_i["ionization_electrons"]
            )
        ]

        electrons.append(
            GeantElectron(
                id=int(electron_id),
                incidence=incidence_i,
                pixels=PixelSet(pixels_i),
                grid=grid,
                trajectory=None,
            )
        )

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
