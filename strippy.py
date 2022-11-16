import argparse
import logging
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


from bruker import load_bruker
from peaks import PeakList
from plotting import even_divide, plot_strips, plot_spectrum_2d
from spectrum import Spectrum

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging._nameToLevel["INFO"])

LONG_DESCRIPTION = """
Make a strip plot from a 3D Bruker data set and 2D peak list. 
Typical usage:
python strippy.py -d 14/pdata/1 -p peaks.list -w 0.15 -c 0.1 10 10 -r 0.5 6.0. 
This would plot processed data 3 from dataset 14.
The peak list has columns 'f3 f2' separated by spaces.
The strips will have width 0.15 ppm.
The contour levels are plotted from 0.1 to 10 with 10 levels total,
each contour has i**(2**0.5) scaling, and spectrum values are normalised by the
standard deviation.
Plotting in the f1 dimension is from 0.5 to 6.0 ppm.
"""

HELP_DIRECTORY = "Directory of Bruker 3D processed data. For example ./1/proc/1/. The folder must contain a 3rr spectrum file with associated status process parameters 'procs'. It is possible to specify multiple directories, in which case they will be plotted over one another."

HELP_PEAKS = "2 dimensional peak list with 'f3 f2' in ppm separated by spaces or tabs. Each row is a single peak like '8.45 120.2'. However, a peak label can be inserted in the first column like '34GLU 8.45 120.2'"

HELP_AXISORDER = "The order of axes for the dataset, default: 0 1 2 for axes f1, f2, f3. You may want to modify this parameter to select which dimension is plotted in x, y and z for the strips. For example, the spectrum with f3=1H, f2=14N and f1=13C will plot by default with x=1H, y=14N, z=13C. However, but choosing axis order '1 0 2' you can have x=1H, y=13C and z=14N. Note that 3 integers are required for each additional dataset provided."

HELP_CONTOURS = "3 numbers specifying the contour levels: 'max min number_levels'. Note all values must be positive, negative contours are automatically generated using the same values. WARNING CURRENTLY IGNORED"

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        prog="Strippy",
        description=LONG_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help=HELP_DIRECTORY,
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument("-p", "--peaks", help=HELP_PEAKS, type=str, required=True)
    parser.add_argument(
        "-a",
        "--axisorder",
        help=HELP_AXISORDER,
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Single strip width in ppm",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "-r",
        "--range",
        help="Minimum and maximum strip ranges in ppm to be plotted in f1 dimension. Two values must be given for each dataset specified",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-i",
        "--dimensions",
        help="Strip dimensions in mm. Two numbers for width and height must be given.",
        type=float,
        nargs=2,
        default=(10.0, 300.0),
    )
    parser.add_argument(
        "-c",
        "--contours",
        help=HELP_CONTOURS,
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-j",
        "--pages",
        help="The number of pages in final pdf.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-l",
        "--logging",
        help="Set the logging level to one of ERROR, WARNING, INFO or DEBUG for increasing levels of information.",
        type=str,
        default="DEBUG",
    )
    args = parser.parse_args()
else:
    args = None


if args:
    # Setup the logger
    print("Setting up log file.")
    logger = logging.getLogger(__name__)
    if args.logging not in logging._nameToLevel:
        raise KeyError("Logging level name not supported")
    logger.setLevel(logging._nameToLevel[args.logging.upper()])
    logging.basicConfig(filename="strippy.log", level=args.logging.upper())

    # Load peaks
    print("Loading peaks.")
    if args.peaks is None:
        raise ValueError("A peak list must be specified")
    peaklist = PeakList.load_from_file(args.peaks)
    logger.info(f"Loaded {len(peaklist)} peaks from file {args.peaks}")

    # Load spectra
    print("Loading spectra.")
    if args.dataset is None:
        logger.error("No dataset specified")
        raise ValueError("A dataset must be specified.")
    if args.axisorder is None:
        axisorder = (0, 1, 2) * len(args.dataset)
        logger.info(f"No axis order was specified. Using default: {axisorder}")
    else:
        axisorder = args.axisorder
    if len(axisorder) != 3 * len(args.dataset):
        logger.error(f"Number of axes in axis order was incorrect: {axisorder}")
        raise IndexError(
            f"Incorrect number of axis order arguments. Should be {3 * len(args.dataset)} but got {len(axisorder)}."
        )
    if args.range is not None:
        if len(args.range) != 2 * len(args.dataset):
            logger.error
            raise IndexError(
                f"Incorrect number of axis range arguments. Should be {2 * len(args.dataset)} but got {len(args.range)}."
            )

    spectra: list[Spectrum] = []
    for i, dataset_path in enumerate(args.dataset):
        spectrum = load_bruker(dataset_path)
        spectrum.reorder_axes(axisorder[i * 3 : i * 3 + 3])
        spectra.append(spectrum)
        logger.info(f"Loaded spectrum from file {dataset_path}: {str(spectrum)}")

    # Slice spectra
    print("Slicing spectra.")
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
            # logger.debug(f"Created new strip for peak {peak} given by {strip}")

        strips_set.append(strips)

        # Create labels
        label = f"{peak.label}: {f2:.2f}"
        labels.append(label)

    # Plot strips
    print("Plotting strips.")
    figs = []
    strips_divided = even_divide(strips_set, args.pages)
    labels_divided = even_divide(labels, args.pages)
    for strips_subset, labels_subset in zip(strips_divided, labels_divided):
        fig = plot_strips(strips_subset, labels_subset)
        figs.append(fig)

    # Set axis ranges
    print("Setting axis ranges.")
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

    # Adjust figure size
    print("Adjusting axis dimensions.")
    width = args.dimensions[0] / 25.4
    height = args.dimensions[1] / 25.4
    for fig in figs:
        naxes = len(fig.axes) // len(spectra)
        fig_width = (width * naxes) / (fig.subplotpars.right - fig.subplotpars.left)
        fig_height = height / (fig.subplotpars.top - fig.subplotpars.bottom)
        fig.set_size_inches(fig_width, fig_height)

    # Save plot to file
    with PdfPages("strips.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")

    # Create projection about z axis
    projection = spectra[0].projection(0)
    fig = plot_spectrum_2d(projection, peaklist)
    fig.savefig("projection.pdf", bbox_inches="tight")
