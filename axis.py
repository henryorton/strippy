"""A class for managing a ppm scale"""
import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class Axis:
    """
    Stores information about the axis ppm values.
    Can provide unit conversion between points, ppm and Hz values.
    The Axis must be ordered with values left to right.

    points : number of points along the axis
    carrier : centre frequency of the axis in Hz
    spectral_width : spectral width of the axis in Hz
    observed_frequency : NMR spectrometer frequency in MHz
    label : nucleus identifier e.g. 1H, 15N, 13C
    interpolate : if False, returned indices are integers rounded to the nearest index
        if True, returned indices are floats representing a continuous index along the axis
    """

    points: int
    carrier: float
    spectral_width: float
    observed_frequency: float
    label: str
    interpolate: bool = False

    def __str__(self):
        return f"<Axis: {self.label}, {self.points} points in {self.ppm_limits} ppm>"

    def __post_init__(self):
        logging.debug(f"Created new axis: {self}")

    @property
    def hz_scale(self) -> npt.NDArray[np.float_]:
        """
        Get an array of Hz values for the axis
        """
        return np.linspace(
            self.carrier + 0.5 * self.spectral_width,
            self.carrier - 0.5 * self.spectral_width,
            self.points,
        )

    @property
    def centre_ppm(self) -> float:
        """The center frequency in ppm"""
        return 0.5 * sum(self.ppm_limits)

    @property
    def ppm_scale(self) -> npt.NDArray[np.float_]:
        """
        Get an array of ppm values for the axis
        """
        return self.hz_scale / self.observed_frequency

    @property
    def ppm_limits(self) -> tuple[float, float]:
        """
        the left and right ppm limits respectively
        """
        lhs = (self.carrier + 0.5 * self.spectral_width) / self.observed_frequency
        rhs = (self.carrier - 0.5 * self.spectral_width) / self.observed_frequency
        return lhs, rhs

    def f(self, ppm: float) -> float:
        """
        Unit conversion from ppm -> index location on the axis.
        This returns the continuous decimal index along the scale

        ppm : the desired ppm value to be translated
        """
        hzpp = self.spectral_width / float(self.points - 1)
        loc = (
            -ppm * self.observed_frequency + self.carrier + self.spectral_width / 2
        ) / hzpp
        return loc

    def i(self, ppm: float) -> int:
        """
        Unit conversion from ppm -> index on the axis.
        This returns the closest integer index along the scale

        ppm : the desired ppm value to be translated
        """
        return round(self.f(ppm))

    def ppm_within_limits(self, ppm: float) -> bool:
        """
        Determine if ppm value is within limits of the axis.

        ppm : the ppm value to be tested.
        """
        lhs, rhs = self.ppm_limits
        if lhs > rhs:
            if lhs >= ppm >= rhs:
                return True
        else:
            if lhs <= ppm <= rhs:
                return True
        return False

    def __getitem__(self, key: slice | float) -> slice | int | float:
        """
        Translate between ppm slice and integer slice
        A slice represents are range of values between <start> and <stop>

        key : The ppm range to be translated.
            If [8.5:4.7] was provided, a new slice containing the
            point ranges would be returned e.g. [55:560].
            If a float is provided, the index along the axis is returned.
            This index could be a rounded int or a float, depnding on
            the interpolation bool
        """

        if isinstance(key, slice):
            if key.start is not None:
                if not self.ppm_within_limits(key.start):
                    raise IndexError(
                        f"The value {key.start} ppm is outside the {self} limits of {self.ppm_limits}"
                    )
                start = key.start
            else:
                start = self.ppm_limits[0]
            if key.stop is not None:
                if not self.ppm_within_limits(key.stop):
                    raise IndexError(
                        f"The value {key.stop} ppm is outside the {self} limits of {self.ppm_limits}"
                    )
                stop = key.stop
            else:
                stop = self.ppm_limits[1]

            lhs, rhs = self.ppm_limits
            if (lhs > rhs) != (start > stop):
                raise IndexError(
                    f"Slice order [{start}:{stop}] ppm is incorrect for {self} which has left to right order {self.ppm_limits} ppm"
                )

            if self.interpolate:
                return slice(self.f(start), self.f(stop) + 1)
            else:
                return slice(self.i(start), self.i(stop) + 1)
        elif isinstance(key, float):
            if not self.ppm_within_limits(key):
                raise IndexError(
                    f"The value {key} ppm is outside the {self} limits of {self.ppm_limits}"
                )
            if self.interpolate:
                return self.f(key)
            else:
                return self.i(key)
        else:
            raise TypeError("Must use a slice or float type in Axis getitem")

    def new(self, key: slice | float) -> Union["Axis", int, float]:
        """
        Return a new instance of the current axis constructed from a subset
        of the axis values.

        key: The ppm range to be taken for the new axis.
            If a float is provided, only the index for that point is returned.
            That index could be an int or float depnding on the rounding settings
        """
        if isinstance(key, slice):
            new_slice = self[key]
            assert isinstance(new_slice, slice)
            scale = self.hz_scale[new_slice]
            return self.__class__(
                points=len(scale),
                carrier=0.5 * (scale.min() + scale.max()),
                spectral_width=float(np.ptp(scale)),
                observed_frequency=self.observed_frequency,
                label=self.label,
            )
        elif isinstance(key, float):
            value = self[key]
            if self.interpolate:
                assert isinstance(value, float)
                return value
            else:
                assert isinstance(value, int)
                return value
        raise TypeError("Must use a slice or float type")


# a = Axis(
#     points=1024,
#     carrier=4.7 * 800.0,
#     spectral_width=10.0 * 800.0,
#     observed_frequency=800.0,
#     label="1H",
# )

# print(a[2.0:8.0])
