import logging
import os
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from axis import Axis
from spectrum import Spectrum

logger = logging.getLogger(__name__)
logger.setLevel(logging._nameToLevel["INFO"])


def reorder_submatrix(
    mangled_data: npt.NDArray,
    final_shape: tuple[int, ...],
    submatrix_shape: tuple[int, ...],
) -> npt.NDArray[np.float_]:
    """
    Reorder processed binary Bruker data to remove submatrix order
    Returns an ordered array with correct shape

    mangled_data : the one dimensional data array with submatrix index ordering
    final_shape : the final shape of the array, e.g. (128, 256, 1024)
    submatrix_shape : the shape of the submatrix
    """
    logging.debug(
        (
            f"Started submatrix reordering from shape {mangled_data.shape} "
            f"with submatrices {submatrix_shape} to give shape {final_shape}"
        )
    )

    if len(submatrix_shape) == 1 or len(final_shape) == 1:
        return mangled_data

    data = np.empty(final_shape, dtype=mangled_data.dtype)
    submatrices_per_dimension = [
        int(i / j) for i, j in zip(final_shape, submatrix_shape)
    ]
    total_submatrices = np.product(submatrices_per_dimension)
    data_submatrix_shaped = mangled_data.reshape(
        [total_submatrices] + list(submatrix_shape)
    )

    for submatrix_number, submatrix_index in enumerate(
        np.ndindex(tuple(submatrices_per_dimension))
    ):
        submatrix_slices = [
            slice(i * j, (i + 1) * j) for i, j in zip(submatrix_index, submatrix_shape)
        ]
        data[tuple(submatrix_slices)] = data_submatrix_shaped[submatrix_number]

    return data


def read_procs(file_name: str) -> dict[str, Any]:
    """
    Read Bruker processed data status parameters from file
    Returns dictionary of key value pairs
    """
    logging.info(f"Reading Bruker processed data {file_name}")

    line_match = re.compile(r"\#\#\$(.*)=[\s+]?(.*)")
    value_match = re.compile("<(.*)>")
    d = {}
    with open(file_name) as f:
        for line in f:
            m = line_match.match(line)
            if m:
                key, value = m.groups()
                v = value_match.match(value)
                if v:
                    d[key] = v.group(1)
                else:
                    try:
                        f = float(value)
                        i = int(f)
                        if f == i:
                            d[key] = i
                        else:
                            d[key] = f
                    except ValueError:
                        d[key] = value
    logging.debug(f"Parsed Bruker processed data: {d}")
    return d


def read_clevels(file_name: str) -> npt.NDArray[np.float_]:
    """
    Read Bruker contour levels from file
    """
    logging.info(f"Reading Bruker contour levels: {file_name}")

    with open(file_name) as f:
        s = f.read().replace("\n", " ")
    match = re.search(r"\#\#\$LEVELS=\s?\(.*\)(.*?)\#\#\$", s)
    if match:
        lvls = match.group(1)
        contours = np.trim_zeros(np.array(lvls.split(), dtype=float))
        logging.debug(f"Parsed Bruker contour levels: {contours}")
        return contours
    else:
        raise KeyError(f"Contour levels were not found in file: {file_name}")


def read_spectrum(file_name: str) -> npt.NDArray[np.int_]:
    """
    Read Bruker spectrum intensities from file
    Can read 1r, 2rr and 3rrr files
    Data is integer type.
    WARNING submatrix ordering may be present
    """
    logging.info(f"Reading Bruker spectrum data file: {file_name}")

    with open(file_name, "rb") as o:
        return np.frombuffer(o.read(), dtype="<i4")


def load_bruker(spectrum_directory: str) -> Spectrum:
    """
    Read a Bruker spectrum from file into Spectrum object
    """
    spectrum_path = None
    clevels_path = None
    procs: dict[int, dict[str, Any]] = {}

    for file_name in os.listdir(spectrum_directory):
        path = os.path.join(spectrum_directory, file_name)

        if file_name in ("1r", "2rr", "3rrr", "4rrrr"):
            spectrum_path = path

        elif re.search("proc[0-9]?s", file_name):
            match = re.search("proc([0-9])?s", file_name)
            if match:
                num = match.group(1)
            else:
                continue
            if not num:
                dimension = 1
            else:
                dimension = int(num)
            procs[int(dimension)] = read_procs(path)

        elif re.search("clevels", file_name):
            clevels_path = path

    if spectrum_path is None:
        raise FileNotFoundError("Cannot locate spectrum data file 1r, 2rr or 3rrr")

    axes = []
    final_shape = []
    submatrix_shape = []
    for dimension in sorted(procs):
        proc = procs[dimension]
        axis = Axis(
            points=proc["SI"],
            carrier=proc["OFFSET"] * proc["SF"] - 0.5 * proc["SW_p"],
            spectral_width=proc["SW_p"],
            observed_frequency=proc["SF"],
            label=proc["AXNUC"],
        )
        axes.append(axis)
        final_shape.append(axis.points)
        submatrix_shape.append(proc.get("XDIM", axis.points))

    scale = 2.0 ** (-procs[1].get("NC_proc", 0.0))

    # Read spectrum data and scale
    data = read_spectrum(spectrum_path)
    data = (
        reorder_submatrix(
            mangled_data=data,
            final_shape=tuple(final_shape[::-1]),
            submatrix_shape=tuple(submatrix_shape[::-1]),
        )
        / scale
    )

    # Read contour levels
    if clevels_path:
        contours = read_clevels(clevels_path)
    else:
        contours = np.array([])

    return Spectrum(data=data, axes=tuple(axes[::-1]), contours=contours)
