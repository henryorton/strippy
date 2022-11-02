from dataclasses import dataclass
from typing import Optional


@dataclass
class Peak:
    label: str
    position: tuple[float, float]
    label_color: Optional[str] = None


@dataclass
class PeakList:
    peaks: list[Peak]

    @classmethod
    def load_from_file(cls, file_name: str) -> "PeakList":
        """
        Parse peaks file to list of tuples that specify an identifier and
        peak position in ppm. If 3 columns are provided, the first column is
        assumed to be the peak identifier (such as a sequence/residue). If two
        columns are provided, the peak identifier is set to an integer.

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
                            float(splt[-1]),
                            float(splt[-2]),
                        ),
                    )
                    peak_list.peaks.append(peak)

                except ValueError:
                    print(
                        f"There was an error when parsing the peaks file on line {line_number+1}."
                    )
                    print(f"This line was ignored when parsing: {repr(line)}")

        return peak_list

    def reverse_order(self):
        for peak in self.peaks:
            peak.position = peak.position[::-1]

        #     cm = plt.get_cmap("brg", len(peaks))
        #     for i, peak in enumerate(peaks):
        #         peak += [cm(i)]
        # return peaks
