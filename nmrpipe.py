import struct

import numpy as np

from axis import Axis
from spectrum import Spectrum

PIPE_HEADER_KEY = [
    ("FDMAGIC", 0, 4, "f"),
    ("FDFLTFORMAT", 4, 8, "f"),
    ("FDFLTORDER", 8, 12, "f"),
    ("FDDIMCOUNT", 36, 40, "i"),
    ("FDF3OBS", 40, 44, "f"),
    ("FDF3SW", 44, 48, "f"),
    ("FDF3ORIG", 48, 52, "f"),
    ("FDF3FTFLAG", 52, 56, "i"),
    ("FDPLANELOC", 56, 60, "f"),
    ("FDF3SIZE", 60, 64, "i"),
    ("FDF2LABEL", 64, 72, "s"),
    ("FDF1LABEL", 72, 80, "s"),
    ("FDF3LABEL", 80, 88, "s"),
    ("FDF4LABEL", 88, 96, "s"),
    ("FDDIMORDER1", 96, 100, "i"),
    ("FDDIMORDER2", 100, 104, "i"),
    ("FDDIMORDER3", 104, 108, "i"),
    ("FDDIMORDER4", 108, 112, "i"),
    ("FDF4OBS", 112, 116, "f"),
    ("FDF4SW", 116, 120, "f"),
    ("FDF4ORIG", 120, 124, "f"),
    ("FDF4FTFLAG", 124, 128, "i"),
    ("FDF4SIZE", 128, 132, "i"),
    ("FDDMXVAL", 160, 164, "f"),
    ("FDDMXFLAG", 164, 168, "i"),
    ("FDDELTATR", 168, 172, "f"),
    ("FDNUSDIM", 180, 184, "i"),
    ("FDF3APOD", 200, 204, "f"),
    ("FDF3QUADFLAG", 204, 208, "i"),
    ("FDF4APOD", 212, 216, "f"),
    ("FDF4QUADFLAG", 216, 220, "i"),
    ("FDF1QUADFLAG", 220, 224, "i"),
    ("FDF2QUADFLAG", 224, 228, "i"),
    ("FDPIPEFLAG", 228, 232, "i"),
    ("FDF3UNITS", 232, 236, "f"),
    ("FDF4UNITS", 236, 240, "f"),
    ("FDF3P0", 240, 244, "f"),
    ("FDF3P1", 244, 248, "f"),
    ("FDF4P0", 248, 252, "f"),
    ("FDF4P1", 252, 256, "f"),
    ("FDF2AQSIGN", 256, 260, "f"),
    ("FDPARTITION", 260, 264, "f"),
    ("FDF2CAR", 264, 268, "f"),
    ("FDF1CAR", 268, 272, "f"),
    ("FDF3CAR", 272, 276, "f"),
    ("FDF4CAR", 276, 280, "f"),
    ("FDUSER1", 280, 284, "f"),
    ("FDUSER2", 284, 288, "f"),
    ("FDUSER3", 288, 292, "f"),
    ("FDUSER4", 292, 296, "f"),
    ("FDUSER5", 296, 300, "f"),
    ("FDPIPECOUNT", 300, 304, "i"),
    ("FDUSER6", 304, 308, "f"),
    ("FDFIRSTPLANE", 308, 312, "i"),
    ("FDLASTPLANE", 312, 316, "i"),
    ("FDF2CENTER", 316, 320, "i"),
    ("FDF1CENTER", 320, 324, "i"),
    ("FDF3CENTER", 324, 328, "i"),
    ("FDF4CENTER", 328, 332, "i"),
    ("FDF2APOD", 380, 384, "f"),
    ("FDF2FTSIZE", 384, 388, "i"),
    ("FDREALSIZE", 388, 392, "i"),
    ("FDF1FTSIZE", 392, 396, "i"),
    ("FDSIZE", 396, 400, "i"),
    ("FDF2SW", 400, 404, "f"),
    ("FDF2ORIG", 404, 408, "f"),
    ("FDQUADFLAG", 424, 428, "i"),
    ("FDF2ZF", 432, 436, "f"),
    ("FDF2P0", 436, 440, "f"),
    ("FDF2P1", 440, 444, "f"),
    ("FDF2LB", 444, 448, "f"),
    ("FDF2OBS", 476, 480, "f"),
    ("FDMCFLAG", 540, 544, "i"),
    ("FDF2UNITS", 608, 612, "f"),
    ("FDNOISE", 612, 616, "f"),
    ("FDTEMPERATURE", 628, 632, "f"),
    ("FDPRESSURE", 632, 636, "f"),
    ("FDRANK", 720, 724, "f"),
    ("FDTAU", 796, 800, "f"),
    ("FDF3FTSIZE", 800, 804, "i"),
    ("FDF4FTSIZE", 804, 808, "i"),
    ("FDF1OBS", 872, 876, "f"),
    ("FDSPECNUM", 876, 880, "i"),
    ("FDF2FTFLAG", 880, 884, "i"),
    ("FDTRANSPOSED", 884, 888, "i"),
    ("FDF1FTFLAG", 888, 892, "i"),
    ("FDF1SW", 916, 920, "f"),
    ("FDF1UNITS", 936, 940, "f"),
    ("FDF1LB", 972, 976, "f"),
    ("FDF1P0", 980, 984, "f"),
    ("FDF1P1", 984, 988, "f"),
    ("FDMAX", 988, 992, "f"),
    ("FDMIN", 992, 996, "f"),
    ("FDF1ORIG", 996, 1000, "f"),
    ("FDSCALEFLAG", 1000, 1004, "i"),
    ("FDDISPMAX", 1004, 1008, "f"),
    ("FDDISPMIN", 1008, 1012, "f"),
    ("FDPTHRESH", 1012, 1016, "f"),
    ("FDNTHRESH", 1016, 1020, "f"),
    ("FD2DPHASE", 1024, 1028, "f"),
    ("FDF2X1", 1028, 1032, "i"),
    ("FDF2XN", 1032, 1036, "i"),
    ("FDF1X1", 1036, 1040, "i"),
    ("FDF1XN", 1040, 1044, "i"),
    ("FDF3X1", 1044, 1048, "i"),
    ("FDF3XN", 1048, 1052, "i"),
    ("FDF4X1", 1052, 1056, "i"),
    ("FDF4XN", 1056, 1060, "i"),
    ("FDDOMINFO", 1064, 1068, "f"),
    ("FDMETHINFO", 1068, 1072, "f"),
    ("FDHOURS", 1132, 1136, "f"),
    ("FDMINS", 1136, 1140, "f"),
    ("FDSECS", 1140, 1144, "f"),
    ("FDSRCNAME", 1144, 1160, "s"),
    ("FDUSERNAME", 1160, 1176, "s"),
    ("FDMONTH", 1176, 1180, "f"),
    ("FDDAY", 1180, 1184, "f"),
    ("FDYEAR", 1184, 1188, "f"),
    ("FDTITLE", 1188, 1248, "s"),
    ("FDCOMMENT", 1248, 1408, "s"),
    ("FDLASTBLOCK", 1436, 1440, "f"),
    ("FDCONTBLOCK", 1440, 1444, "f"),
    ("FDBASEBLOCK", 1444, 1448, "f"),
    ("FDPEAKBLOCK", 1448, 1452, "f"),
    ("FDBMAPBLOCK", 1452, 1456, "f"),
    ("FDHISTBLOCK", 1456, 1460, "f"),
    ("FD1DBLOCK", 1460, 1464, "f"),
    ("FDSCORE", 1480, 1484, "f"),
    ("FDSCANS", 1484, 1488, "f"),
    ("FDF3LB", 1488, 1492, "f"),
    ("FDF4LB", 1492, 1496, "f"),
    ("FDF2GB", 1496, 1500, "f"),
    ("FDF1GB", 1500, 1504, "f"),
    ("FDF3GB", 1504, 1508, "f"),
    ("FDF4GB", 1508, 1512, "f"),
    ("FDF2OBSMID", 1512, 1516, "f"),
    ("FDF1OBSMID", 1516, 1520, "f"),
    ("FDF3OBSMID", 1520, 1524, "f"),
    ("FDF4OBSMID", 1524, 1528, "f"),
    ("FDF2GOFF", 1528, 1532, "f"),
    ("FDF1GOFF", 1532, 1536, "f"),
    ("FDF3GOFF", 1536, 1540, "f"),
    ("FDF4GOFF", 1540, 1544, "f"),
    ("FDF2TDSIZE", 1544, 1548, "i"),
    ("FDF1TDSIZE", 1548, 1552, "i"),
    ("FDF3TDSIZE", 1552, 1556, "i"),
    ("FDF4TDSIZE", 1556, 1560, "i"),
    ("FD2DVIRGIN", 1596, 1600, "f"),
    ("FDF3APODCODE", 1600, 1604, "f"),
    ("FDF3APODQ1", 1604, 1608, "f"),
    ("FDF3APODQ2", 1608, 1612, "f"),
    ("FDF3APODQ3", 1612, 1616, "f"),
    ("FDF3C1", 1616, 1620, "f"),
    ("FDF4APODCODE", 1620, 1624, "f"),
    ("FDF4APODQ1", 1624, 1628, "f"),
    ("FDF4APODQ2", 1628, 1632, "f"),
    ("FDF4APODQ3", 1632, 1636, "f"),
    ("FDF4C1", 1636, 1640, "f"),
    ("FDF2APODCODE", 1652, 1656, "f"),
    ("FDF1APODCODE", 1656, 1660, "f"),
    ("FDF2APODQ1", 1660, 1664, "f"),
    ("FDF2APODQ2", 1664, 1668, "f"),
    ("FDF2APODQ3", 1668, 1672, "f"),
    ("FDF2C1", 1672, 1676, "f"),
    ("FDF2APODDF", 1676, 1680, "f"),
    ("FDF1APODQ1", 1680, 1684, "f"),
    ("FDF1APODQ2", 1684, 1688, "f"),
    ("FDF1APODQ3", 1688, 1692, "f"),
    ("FDF1C1", 1692, 1696, "f"),
    ("FDF1APOD", 1712, 1716, "f"),
    ("FDF1ZF", 1748, 1752, "i"),
    ("FDF3ZF", 1752, 1756, "i"),
    ("FDF4ZF", 1756, 1760, "i"),
    ("FDFILECOUNT", 1768, 1772, "i"),
    ("FDSLICECOUNT", 1772, 1776, "i"),
    ("FDSLICECOUNT0", 1772, 1776, "i"),
    ("FDTHREADCOUNT", 1776, 1780, "i"),
    ("FDTHREADID", 1780, 1784, "f"),
    ("FDSLICECOUNT1", 1784, 1788, "i"),
    ("FDCUBEFLAG", 1788, 1792, "i"),
    ("FDOPERNAME", 1856, 1888, "s"),
    ("FDF1AQSIGN", 1900, 1904, "f"),
    ("FDF3AQSIGN", 1904, 1908, "f"),
    ("FDF4AQSIGN", 1908, 1912, "f"),
    ("FDF2OFFPPM", 1920, 1924, "f"),
    ("FDF1OFFPPM", 1924, 1928, "f"),
    ("FDF3OFFPPM", 1928, 1932, "f"),
    ("FDF4OFFPPM", 1932, 1936, "f"),
]


def load_pipe(spectrum_file: str) -> Spectrum:
    """Load NMRPipe spectrum from file"""
    with open(spectrum_file, "rb") as buff:
        rawHdr = buff.read(2048)
        data = np.fromfile(buff, "float32")

    hdr = {}
    for label, start, end, dtype in PIPE_HEADER_KEY:
        rawBytes = rawHdr[start:end]
        if dtype in ("f", "i"):
            formBytes = "{}{}".format((end - start) // 4, "f")
        else:
            formBytes = "{}{}".format(end - start, dtype)
        value = struct.unpack(formBytes, rawBytes)[0]
        if dtype == "s":
            value = value.strip(b"\x00").decode()
        if dtype == "i":
            value = int(value)
        hdr[label] = value

    axes: list[Axis] = []
    dims: list[int] = []
    for i in range(1, 4):
        dim = hdr["FDDIMORDER{}".format(i)]
        dims.append(dim - 1)
        x1 = hdr["FDF{}X1".format(dim)]
        xn = hdr["FDF{}XN".format(dim)]
        if xn == 0:
            xn = hdr["FDF{}FTSIZE".format(dim)] - 1
        points = xn - x1 + 1
        spectra_width = hdr["FDF{}SW".format(dim)]
        axis = Axis(
            points=points,
            carrier=hdr["FDF{}ORIG".format(dim)] + spectra_width / 2.0,
            spectral_width=spectra_width,
            observed_frequency=hdr["FDF{}OBS".format(dim)],
            label=hdr["FDF{}LABEL".format(dim)],
        )
        axes.append(axis)

    axes = [axes[i] for i in dims]
    data = data.reshape(*[axis.points for axis in axes])
    data /= np.std(data)
    spectrum = Spectrum(
        data=data,
        axes=tuple(axes),
        contours=np.array([]),
    )
    spectrum.make_contours(5, 20, 10)
    return spectrum
