from typing import Optional, TypeVar

import matplotlib as mpl
import numpy as np

from spectrum import Spectrum

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


def plot_spectrum_2d(spectrum: Spectrum):
    pass


def plot_strips(strips_set: list[list[Spectrum]], labels: Optional[list[str]] = None):
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

        strip = strips[0]

        ax.tick_params(right="off")
        f1l, f1r = strip.axes[0].ppm_limits
        if f1l > f1r:
            f1l_round = int(f1l * 2) / 2
            f1r_round = int((f1r + 1) * 2) / 2
            step = -0.5
        else:
            f1l_round = int((f1l + 1) * 2) / 2
            f1r_round = int(f1r * 2) / 2
            step = 0.5
        ax.set_yticks(np.arange(f1l_round, f1r_round + step, step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if i != 0:
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks_position("none")

        ax.set_title(labels[i], rotation=90, verticalalignment="bottom")

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticks([strip.axes[1].centre_ppm])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.yaxis.grid(linestyle="dotted")

    return fig
