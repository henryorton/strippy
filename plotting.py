from typing import Optional, TypeVar

import matplotlib as mpl
import numpy as np
from peaks import PeakList

from spectrum import Spectrum
from matplotlib.figure import Figure

mpl.use("Agg")
import warnings

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore", category=UserWarning)


CONTOUR_COLOURS = [("b", "g"), ("r", "m"), ("k", "o")]


T = TypeVar("T")


def even_divide(lst: list[T], n: int) -> list[list[T]]:
    """Divide a list into n roughly even chunks"""
    p = len(lst) // n
    if len(lst) - p > 0:
        return [lst[:p]] + even_divide(lst[p:], n - 1)
    else:
        return [lst]


def plot_spectrum_1d(spectrum: Spectrum):
    pass


def plot_spectrum_2d(spectrum: Spectrum, peaklist: PeakList) -> Figure:
    fig = plt.figure(figsize=(16.5, 11.7))
    ax = fig.add_subplot(111)

    if spectrum.positive_contours is not None:
        ax.contour(
            spectrum.data,
            spectrum.positive_contours,
            colors="b",
            extent=spectrum.extent,
            linewidths=0.05,
        )
    if spectrum.negative_contours is not None:
        ax.contour(
            spectrum.data,
            spectrum.negative_contours,
            colors="g",
            extent=spectrum.extent,
            linewidths=0.05,
        )
    for peak in peaklist:
        ax.plot(*peak.position, color="r", marker="x")
        ax.annotate(peak.label, xy=peak.position, fontsize=5)

    ax.invert_xaxis()
    ax.invert_yaxis()

    return fig


def plot_strips(
    strips_set: list[list[Spectrum]], labels: Optional[list[str]] = None
) -> Figure:
    """
    Plot a list of spectra in a strip plot

    strips_set : the outer list is the strips,
        the inner list is the spectra to be plotted on a single strip

    labels : the axis title for each strip
    """
    if labels is None:
        labels = [""] * len(strips_set)
    assert len(strips_set) == len(labels)

    fig = plt.figure(figsize=(16.5, 11.7))
    fig.subplots_adjust(wspace=0)

    for i, strips in enumerate(strips_set):
        ax = fig.add_subplot(1, len(strips_set), i + 1)
        for strip, color in zip(strips, CONTOUR_COLOURS):

            if len(strip.positive_contours) > 0:
                ax.contour(
                    strip.data,
                    levels=strip.positive_contours,
                    extent=strip.extent,
                    colors=color[0],
                    linewidths=0.05,
                )
            if len(strip.negative_contours) > 0:
                ax.contour(
                    strip.data,
                    levels=strip.negative_contours,
                    extent=strip.extent,
                    colors=color[1],
                    linewidths=0.05,
                )

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.tick_params(right="off")
        if i != 0:
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks_position("none")

        ax.set_title(labels[i], rotation=90, verticalalignment="bottom")

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticks([strips[0].axes[1].centre_ppm])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.yaxis.grid(linestyle="dotted")

    return fig
