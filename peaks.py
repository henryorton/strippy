from dataclasses import dataclass
from typing import Optional

from matplotlib import colormaps


@dataclass
class Peak:
    label: str
    position: tuple[float, float]
    label_color: Optional[str] = None


@dataclass
class PeakList:
    peaks: list[Peak]

    def __iter__(self):
        for peak in self.peaks:
            yield peak

    def __len__(self):
        return len(self.peaks)

    @classmethod
    def load_from_file(cls, file_name: str) -> "PeakList":
        """
        Parse peaks file to list of tuples that specify an identifier and
        peak position in ppm. If 3 columns are provided, the first column is
        assumed to be the peak identifier (such as a sequence/residue). If two
        columns are provided, the peak identifier is set to an integer.
        Lines starting with # are ignored

        Parameters
        ----------
        fileName : str
            the directory and filename of the peaks file to be loaded

        Returns
        -------
        peaks : list of tuples
            a list of tuples (i, (f3, f2)) where i is an identifier, f3 and f2
            make a numpy array of the ppm positions of the peak.
        """
        peak_list = cls(peaks=[])
        with open(file_name) as o:
            index = 0
            for line_number, line in enumerate(o):
                if line.startswith("#"):
                    continue
                splt = line.split()
                try:
                    # There is no label specified
                    if len(splt) == 2:
                        label = str(index)
                        index += 1
                    # There is a label specified
                    elif len(splt) == 3:
                        label = splt[0]
                    else:
                        raise ValueError

                    peak = Peak(
                        label=label,
                        position=(
                            float(splt[-2]),
                            float(splt[-1]),
                        ),
                    )
                    peak_list.peaks.append(peak)

                except ValueError:
                    print(
                        f"There was an error when parsing the peaks file on line {line_number+1}."
                    )
                    print(f"This line was ignored when parsing: {repr(line)}")

        peak_list.set_colour_map()
        return peak_list

    def swap_order(self):
        for peak in self.peaks:
            peak.position = peak.position[::-1]

    def set_colour_map(self):

        cm = colormaps["viridis"].resampled(len(self.peaks))
        for i, peak in enumerate(self.peaks):
            peak.label_color = cm(i)
