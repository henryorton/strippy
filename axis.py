"""A class for managing a ppm scale"""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Axis:
    """
    Stores information about the axis ppm values.
    Can provide unit conversion between points, ppm and Hz values.
    Note all scales are returned in descending order

    points : number of points along the axis
    carrier : centre frequency of the axis in Hz
    spectral_width : spectral width of the axis in Hz
    observed_frequency : acquired NMR spectrometer frequency in MHz
    dimension : dimension of the current axis
    label : nucleus identifier e.g. 1H, 15N, 13C
    """

    points: int
    carrier: float
    spectral_width: float
    observed_frequency: float
    dimension: int
    label: str

    def __str__(self):
        return f"<Axis {self.dimension}:{self.label:>3}>"

    def __repr__(self):
        return self.__str__()

    @property
    def hz_scale(self) -> npt.NDArray[np.float_]:
        """
        Returns
        -------
        array : numpy.ndarray of floats
                the axis values in Hz in descending order
        """
        return np.linspace(
            self.carrier + 0.5 * self.spectral_width,
            self.carrier - 0.5 * self.spectral_width,
            self.points,
        )

    @property
    def ppm_scale(self) -> npt.NDArray[np.float_]:
        """
        Returns
        -------
        array : numpy.ndarray of floats
                the axis values in ppm in descending order
        """
        return self.hz_scale / self.observed_frequency

    @property
    def ppm_limits(self) -> tuple[float, float]:
        """
        Returns
        -------
        tuple : (float, float)
                the high and low ppm limits respectively
        """
        lo = (self.carrier + 0.5 * self.spectral_width) / self.observed_frequency
        hi = (self.carrier - 0.5 * self.spectral_width) / self.observed_frequency
        return lo, hi

    def f(self, ppm: float) -> float:
        """
        Unit conversion from ppm -> points location on the axis.
        This returns the decimal location along the scale

        Parameters
        ----------
        ppm : float

        Returns
        -------
        point : float
                the decimal point location along the axis
        """
        hzpp = self.spectral_width / float(self.points - 1)
        loc = (
            -ppm * self.observed_frequency + self.carrier + self.spectral_width / 2
        ) / hzpp
        return loc

    def i(self, ppm: float) -> int:
        """
        Unit conversion from ppm -> points location on the axis.
        This returns the closest integer location along the scale

        Parameters
        ----------
        ppm : float

        Returns
        -------
        point : int
                the integer point location closest the the provided ppm value
        """
        return int(np.argmin(np.abs(self.ppm_scale - ppm)))

    def __getitem__(self, slic: slice | float) -> slice | int:
        """
        This magic method offers immediate translation between ppm and point
        ranges.

        Parameters
        ----------
        slic : slice or float
                The ppm range to be translated. If [8.5:4.7] was provided, a new
                slice containing the point ranges would be returned e.g. [55:560].
                If a float is provided, the integer point location closest to
                that point is returned.

        Returns
        -------
        point range : slice or int
                the integer range or point closes to the given ppm range or point
                respectively
        """
        if isinstance(slic, slice):
            if slic.start:
                start = slic.start
            else:
                start = self.ppm_limits[0]
            if slic.stop:
                stop = slic.stop
            else:
                stop = self.ppm_limits[1]
            return slice(self.i(start), self.i(stop) + 1)
        elif isinstance(slic, float):
            return self.i(slic)
        else:
            raise TypeError("Must use a slice or float type in Axis getitem")

    def new(self, slic: slice | float) -> "Axis" | tuple[int, int, float]:
        """
        Return a new instance of the current axis constructed from a subset
        of the axis values.

        Parameters
        ----------
        slic : slice or float
                The ppm range to be taken for the new axis. If a float is provided,
                something else happens

        Returns
        -------
        Axis : object
                the integer range or point closes to the given ppm range or point
                respectively
        """
        if isinstance(slic, slice):
            scale = self.hz_scale[self[slic]]
            return self.__class__(
                points=len(scale),
                carrier=float(np.ptp(scale)),
                spectral_width=0.5 * (scale.min() + scale.max()),
                observed_frequency=self.observed_frequency,
                dimension=self.dimension,
                label=self.label,
            )
        elif isinstance(slic, float):
            loc = self.f(slic)
            low = int(loc)
            high = low + 1
            weight = loc - low
            return low, high, weight
        else:
            raise TypeError("Must use a slice or float type")
