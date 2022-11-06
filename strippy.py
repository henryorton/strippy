import argparse
import logging
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from bruker import load_bruker
from peaks import PeakList
from plotting import even_divide, plot_strips
from spectrum import Spectrum

logger = logging.getLogger()
logger.setLevel(logging._nameToLevel["INFO"])

long_description = """
Make a strip plot from a 3D Bruker data set and 2D peak list. 
Typical usage: 
python strippy.py -d 14/pdata/3 -p peaks.list -w 0.15 -c 0.1 10 10 -r 0.5 6.0. 
This would plot processed data 3 from dataset 14.
The peak list has columns 'f3 f2' separated by spaces.
The strips will have width 0.15 ppm.
The contour levels are plotted from 0.1 to 10 with 10 levels total,
each contour has i**(2**0.5) scaling, and spectrum values are normalised by the
standard deviation.
Plotting in the f1 dimension is from 0.5 to 6.0 ppm.
"""


if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument(
        "-d",
        "--dataset",
        help="directory(s) of Bruker 3D dataset or nmrPipe 3D file(s)",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "-p",
        "--peaks",
        help="2 dimensional peak list with 'f3 f2' in ppm separated by spaces",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="strip width in ppm, optional: default=0.15",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "-c",
        "--contours",
        help="3 numbers specifying the contour levels: 'low high number'\
        note all values must be positive, negative contours are plotted the same",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-r",
        "--range",
        help="Range IN PPM to be plotted in f1 dimension e.g. '6.0, 0.0'",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-s",
        "--hsqc",
        help="directory of 2D HSQC data",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--opposite",
        help="reverse f3,f2 peak order to f2,f3",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--projectionaxis",
        help="the axis about which slices will be taken: x or y or z",
        default="z",
        type=str,
    )
    parser.add_argument(
        "-q",
        "--axisorder",
        help="the order of axes for the dataset, default: 0 1 2",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "-j",
        "--pages",
        help="number of pages in final pdf",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-x",
        "--ccpnpeaks",
        help="ccpnmr type peak list",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
else:
    args = None


if args:
    assert isinstance(args.width, float)

    # Load peaks
    if args.peaks is None:
        raise ValueError("A peak list must be specified")
    peaklist = PeakList.load_from_file(args.peaks)
    logging.info(f"Loaded {len(peaklist)} peaks from file {args.peaks}")

    # Load spectra
    if args.dataset is None:
        raise ValueError("A dataset must be specified.")
    if args.axisorder is None:
        axisorder = (0, 1, 2) * len(args.dataset)
    else:
        axisorder = args.axisorder
    if len(axisorder) != 3 * len(args.dataset):
        raise IndexError(
            f"Incorrect number of axis order arguments. Should be {3 * len(args.dataset)} but got {len(axisorder)}."
        )
    if args.range is not None:
        if len(args.range) != 2 * len(args.dataset):
            raise IndexError(
                f"Incorrect number of axis range arguments. Should be {2 * len(args.dataset)} but got {len(args.range)}."
            )

    spectra: list[Spectrum] = []
    for i, dataset_path in enumerate(args.dataset):
        spectrum = load_bruker(dataset_path)
        spectrum.reorder_axes(axisorder[i * 3 : i * 3 + 3])
        spectra.append(spectrum)
        logging.info(f"Loaded spectrum from file {dataset_path}: {str(spectrum)}")

    # Slice spectra
    strips_set: list[list[Spectrum]] = []
    labels: list[str] = []
    for peak in peaklist:
        f2 = peak.position[1]
        f3l = peak.position[0] + 0.5 * args.width
        f3r = peak.position[0] - 0.5 * args.width

        strips: list[Spectrum] = []
        for i, spectrum in enumerate(spectra):
            if args.range:
                f1 = slice(*args.range[i * 2 : i * 2 + 2])
            else:
                f1 = slice(None)
            strip = spectrum[f1, f2, f3l:f3r]
            strips.append(strip)
            logger.info(f"Created new strip for peak {peak} given by {strip}")

        strips_set.append(strips)

        # Create labels
        label = f"{peak.label}: {f2:.2f}"
        labels.append(label)

    # Plot strips
    figs = []
    strips_divided = even_divide(strips_set, args.pages)
    labels_divided = even_divide(labels, args.pages)
    for strips_subset, labels_subset in zip(strips_divided, labels_divided):
        fig = plot_strips(strips_subset, labels_subset)
        figs.append(fig)

    # Set axis ranges
    for fig in figs:
        for ax in fig.axes:
            f1l, f1r = ax.get_ylim()
            if f1l > f1r:
                f1l_round = int(f1l * 2) / 2
                f1r_round = int((f1r + 1) * 2) / 2
                step = -0.5
            else:
                f1l_round = int((f1l + 1) * 2) / 2
                f1r_round = int(f1r * 2) / 2
                step = 0.5
            ax.set_yticks(np.arange(f1l_round, f1r_round + step, step))

    # Save plot to file
    strips_file_name = "strips.pdf"
    with PdfPages(strips_file_name) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


#     if args.hsqc:
#         print("Plotting HSQC ...")
#         hsqc = Spectrum.load_bruker(args.hsqc)
#     else:
#         print("Plotting projection ...")
#         hsqc = spec.projection(projAxis)

#     fig = plt.figure(figsize=(16.5, 11.7))
#     ax = fig.add_subplot(111)

#     if spec.poscont is not None:
#         ax.contour(
#             hsqc.data, hsqc.poscont, colors="b", extent=hsqc.extent, linewidths=0.05
#         )
#     if spec.negcont is not None:
#         ax.contour(
#             hsqc.data, hsqc.negcont, colors="g", extent=hsqc.extent, linewidths=0.05
#         )
#     for lbl, peak, lblcol in peaks:
#         ax.plot(*peak, color="r", marker="x")
#         ax.annotate(lbl, xy=peak, color=lblcol, fontsize=5)

#     ax.invert_xaxis()
#     ax.invert_yaxis()
#     hsqcFileName = "hsqc.pdf"
#     fig.savefig(hsqcFileName, bbox_inches="tight")
#     print("{} file written".format(hsqcFileName))
