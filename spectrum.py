"""A container for spectrum data with axes"""
from dataclasses import dataclass
from axis import Axis
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)


@dataclass
class Spectrum:
    """
    Stores spectrum data matrix and axes info.

    data : a 1, 2 or 3 dimensional array of intensity values
    axes : the axis scales for each dimension of the data
    contours : the countour intensity levels to draw when plotting
    """

    data: npt.NDArray[np.float_]
    axes: tuple[Axis, ...]
    contours: npt.NDArray[np.float_]

    def __str__(self):
        axes = ", ".join(list(map(str, self.axes)))
        return f"<{len(self.axes)}D spectrum with axes [{axes}]>"

    def __post_init__(self):
        logging.info(f"Created new spectrum: {self}")

    def make_contours(self, minimum: float, maximum: float, number: int):
        """
        Calculate contour levels using a sqrt(2) power ratio.
        The arguments must all be positive. Negative contour levels are automatically
        assigned to match the positive contour levels

        minimum : the minimum contour level
        maximum : the maximum contour level
        number : the number of contour levels
        """
        pos = minimum + (maximum - minimum) * (
            np.arange(0, number) / float(number - 1)
        ) ** (2.0**0.5)
        neg = -pos[::-1]
        self.contours = np.concatenate([neg, pos])

    def __getitem__(self, key: float | slice | tuple[float | slice, ...]) -> "Spectrum":
        """
        Slice the data matrix to return a new spectrum with a subset of the data.
        For example, given a 3D spectrum:
            spectrum[:,:,:] returns the same spectrum
            spectrum[25.0,:,:] returns a 2D slice of the spectrum at 25.0 ppm in the first dimension
            spectrum[:,:,10.0:4.7] returns a 3D spectrum with the third dimension reduced to 10.0 to 4.7 ppm range
            spectrum[25.0,45.1,:] returns a 1D spectrum in the third dimension at 25.0 ppm and 45.1 ppm
                in the first and thrid dimensions
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(self.axes) != len(key):
            raise IndexError(
                f"Slicing spectrum requires {len(self.axes)} dimensions, but {len(key)} were provided"
            )

        new_slice = []
        new_axes: list[Axis] = []
        for axis, slic in zip(self.axes, key):
            new_slice.append(axis[slic])
            new_axis = axis.new(slic)
            if isinstance(new_axis, Axis):
                new_axes.append(new_axis)
        return self.__class__(
            data=self.data[tuple(new_slice)],
            axes=tuple(new_axes),
            contours=self.contours,
        )

    @property
    def extent(self) -> tuple[float, ...]:
        """
        Get the extent of the spectrum as a tuple of ppm ranges in each dimension
        """
        ex = ()
        for axis in self.axes:
            ex += axis.ppm_limits[::-1]
        return ex[::-1]

    @property
    def positive_contours(self) -> npt.NDArray[np.float_]:
        """Get only the positive contour levels"""
        return self.contours[np.where(self.contours >= 0)]

    @property
    def negative_contours(self) -> npt.NDArray[np.float_]:
        """Get only the negative contour levels"""
        return self.contours[np.where(self.contours < 0)]

    def projection(self, axis_index: int) -> "Spectrum":
        """
        Create a new spectrum that is a skyline projection along the given axis.

        axis_index : the axis that will be projected
        """
        data = np.max(self.data, axis=axis_index)
        axes = [ax for i, ax in enumerate(self.axes) if i != axis_index]
        return self.__class__(
            data=data,
            axes=tuple(axes),
            contours=self.contours,
        )

    def reorder_axes(self, new_axis_order: tuple[int, ...]):
        """
        Reorder the axis dimensions of the current spectrum.

        new_axis_order : the indices of the new axis order
            for example (1,0,2) would map axis 1 -> 0, 0 -> 1 and 2 -> 2.
        """
        allowed_axes = [
            set([0, 1]),
            set([0, 1, 2]),
        ]
        if len(self.axes) == 1:
            raise IndexError("Cannot reorder axes of spectrum with only one dimension")
        if len(set(new_axis_order)) != len(self.axes):
            raise IndexError("Not enough axes were provided")
        if set(new_axis_order) not in allowed_axes:
            raise IndexError("Incompatible axis definitions")

        self.axes = tuple([self.axes[i] for i in new_axis_order])
        original_axis_order = list(range(len(self.axes)))
        self.data = np.moveaxis(self.data, new_axis_order, original_axis_order)


# axes = (
#     Axis(
#         points=256,
#         carrier=100.0 * 200.0,
#         spectral_width=200.0 * 200.0,
#         observed_frequency=200.0,
#         label="13C",
#     ),
#     Axis(
#         points=128,
#         carrier=120.0 * 80.0,
#         spectral_width=50.0 * 80.0,
#         observed_frequency=80.0,
#         label="14N",
#     ),
#     Axis(
#         points=1024,
#         carrier=4.7 * 800.0,
#         spectral_width=10.0 * 800.0,
#         observed_frequency=800.0,
#         label="1H",
#     ),
# )


# data = np.zeros([axis.points for axis in axes])

# s = Spectrum(
#     data=data,
#     axes=axes,
#     contours=np.array([]),
# )

# print(s)
# print(s.data.shape)
# s.reorder_axes((0, 2, 1))
# print(s)
# print(s.data.shape)
