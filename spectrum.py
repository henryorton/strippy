"""A container for spectrum data with axes"""
from dataclasses import dataclass
from axis import Axis
import numpy as np
import numpy.typing as npt


@dataclass
class Spectrum:
    """
    Stores spectrum data matrix and axes info.
    """

    data: npt.NDArray[np.float_]
    axes: tuple[Axis, ...]
    contours: npt.NDArray[np.float_]

    def make_contours(self, lowest: float, highest: float, number: int):
        pos = lowest + (highest - lowest) * (
            np.arange(0, number) / float(number - 1)
        ) ** (2.0**0.5)
        neg = -pos[::-1]
        self.contours = np.concatenate([neg, pos])

    def __getitem__(
        self, slices: float | slice | tuple[float | slice, ...]
    ) -> "Spectrum":
        if not isinstance(slices, tuple):
            slices = (slices,)
        if len(self.axes) != len(slices):
            raise ValueError(
                f"Slicing spectrum requires {len(self.axes)} dimensions, but {len(slices)} were provided"
            )

        new_slice = []
        new_axes: list[Axis] = []
        for axis, slic in zip(self.axes, slices):
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
        ex = ()
        for axis in self.axes:
            ex += axis.ppm_limits[::-1]
        return ex[::-1]

    @property
    def poscont(self) -> npt.NDArray[np.float_]:
        return self.contours[np.where(self.contours >= 0)]

    @property
    def negcont(self) -> npt.NDArray[np.float_]:
        return self.contours[np.where(self.contours < 0)]

    def projection(self, axis_index: int) -> "Spectrum":
        data = np.max(self.data, axis=axis_index)
        axes = [ax for i, ax in enumerate(self.axes) if i != axis_index]
        return self.__class__(
            data=data,
            axes=tuple(axes),
            contours=self.contours,
        )

    def reorder_axes(self, new_axis_order: tuple[int, ...]):
        self.axes = tuple([self.axes[i] for i in new_axis_order])
        for i, axis in enumerate(self.axes):
            axis.dimension = 3 - i
        axis_nums = [i for i, axis in enumerate(self.axes)]
        self.data = np.moveaxis(self.data, axis_nums, new_axis_order)

    def transpose(self):
        new_order = tuple(range(len(self.axes)))[::-1]
        self.reorder_axes((new_order))
