import numpy as np
import os
import sys
import re
import struct
from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import warnings


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


if __name__=='__main__' and len(sys.argv)>1:
	parser = argparse.ArgumentParser(description=long_description)
	parser.add_argument('-d','--dataset',
		help="directory(s) of Bruker 3D dataset or nmrPipe 3D file(s)",type=str,nargs='+')
	parser.add_argument('-p','--peaks',
		help="2 dimensional peak list with 'f3 f2' in ppm separated by spaces",
		type=str)
	parser.add_argument('-w','--width',
		help="strip width in ppm, optional: default=0.15",
		type=float, default=0.15)
	parser.add_argument('-c','--contours',
		help="3 numbers specifying the contour levels: 'low high number'\
		note all values must be positive, negative contours are plotted the same",
		type=float, nargs=3)
	parser.add_argument('-r','--range',
		help="Range IN PPM to be plotted in f1 dimension e.g. '0.0 6.0'",
		type=float, nargs=2)
	parser.add_argument('-s','--hsqc',
		help="directory of nmrPipe 2D HSQC data",
		type=str)
	parser.add_argument('-o','--opposite',
		help="reverse f3,f2 peak order to f2,f3",
		default=False, action="store_true")
	parser.add_argument('-a','--projectionaxis',
		help="the axis about which slices will be taken: x or y or z",
		default='z', type=str)
	parser.add_argument('-q','--axisorder',
		help="the order of axes for the dataset, default: x y z",
		default=('z','y','x'), type=str, nargs=3)
	parser.add_argument('-j','--pages',
		help="number of pages in final pdf",
		type=int)
	parser.add_argument('-x','--ccpnpeaks',
		help="ccpnmr type peak list",
		default=False, action="store_true")
	args = parser.parse_args()
else:
	args = None





pipe_hdr_key = [
('FDMAGIC'      ,0   ,4   ,'f'),
('FDFLTFORMAT'  ,4   ,8   ,'f'),
('FDFLTORDER'   ,8   ,12  ,'f'),
('FDDIMCOUNT'   ,36  ,40  ,'i'),
('FDF3OBS'      ,40  ,44  ,'f'),
('FDF3SW'       ,44  ,48  ,'f'),
('FDF3ORIG'     ,48  ,52  ,'f'),
('FDF3FTFLAG'   ,52  ,56  ,'i'),
('FDPLANELOC'   ,56  ,60  ,'f'),
('FDF3SIZE'     ,60  ,64  ,'i'),
('FDF2LABEL'    ,64  ,72  ,'s'),
('FDF1LABEL'    ,72  ,80  ,'s'),
('FDF3LABEL'    ,80  ,88  ,'s'),
('FDF4LABEL'    ,88  ,96  ,'s'),
('FDDIMORDER1'  ,96  ,100 ,'i'),
('FDDIMORDER2'  ,100 ,104 ,'i'),
('FDDIMORDER3'  ,104 ,108 ,'i'),
('FDDIMORDER4'  ,108 ,112 ,'i'),
('FDF4OBS'      ,112 ,116 ,'f'),
('FDF4SW'       ,116 ,120 ,'f'),
('FDF4ORIG'     ,120 ,124 ,'f'),
('FDF4FTFLAG'   ,124 ,128 ,'i'),
('FDF4SIZE'     ,128 ,132 ,'i'),
('FDDMXVAL'     ,160 ,164 ,'f'),
('FDDMXFLAG'    ,164 ,168 ,'i'),
('FDDELTATR'    ,168 ,172 ,'f'),
('FDNUSDIM'     ,180 ,184 ,'i'),
('FDF3APOD'     ,200 ,204 ,'f'),
('FDF3QUADFLAG' ,204 ,208 ,'i'),
('FDF4APOD'     ,212 ,216 ,'f'),
('FDF4QUADFLAG' ,216 ,220 ,'i'),
('FDF1QUADFLAG' ,220 ,224 ,'i'),
('FDF2QUADFLAG' ,224 ,228 ,'i'),
('FDPIPEFLAG'   ,228 ,232 ,'i'),
('FDF3UNITS'    ,232 ,236 ,'f'),
('FDF4UNITS'    ,236 ,240 ,'f'),
('FDF3P0'       ,240 ,244 ,'f'),
('FDF3P1'       ,244 ,248 ,'f'),
('FDF4P0'       ,248 ,252 ,'f'),
('FDF4P1'       ,252 ,256 ,'f'),
('FDF2AQSIGN'   ,256 ,260 ,'f'),
('FDPARTITION'  ,260 ,264 ,'f'),
('FDF2CAR'      ,264 ,268 ,'f'),
('FDF1CAR'      ,268 ,272 ,'f'),
('FDF3CAR'      ,272 ,276 ,'f'),
('FDF4CAR'      ,276 ,280 ,'f'),
('FDUSER1'      ,280 ,284 ,'f'),
('FDUSER2'      ,284 ,288 ,'f'),
('FDUSER3'      ,288 ,292 ,'f'),
('FDUSER4'      ,292 ,296 ,'f'),
('FDUSER5'      ,296 ,300 ,'f'),
('FDPIPECOUNT'  ,300 ,304 ,'i'),
('FDUSER6'      ,304 ,308 ,'f'),
('FDFIRSTPLANE' ,308 ,312 ,'i'),
('FDLASTPLANE'  ,312 ,316 ,'i'),
('FDF2CENTER'   ,316 ,320 ,'i'),
('FDF1CENTER'   ,320 ,324 ,'i'),
('FDF3CENTER'   ,324 ,328 ,'i'),
('FDF4CENTER'   ,328 ,332 ,'i'),
('FDF2APOD'     ,380 ,384 ,'f'),
('FDF2FTSIZE'   ,384 ,388 ,'i'),
('FDREALSIZE'   ,388 ,392 ,'i'),
('FDF1FTSIZE'   ,392 ,396 ,'i'),
('FDSIZE'       ,396 ,400 ,'i'),
('FDF2SW'       ,400 ,404 ,'f'),
('FDF2ORIG'     ,404 ,408 ,'f'),
('FDQUADFLAG'   ,424 ,428 ,'i'),
('FDF2ZF'       ,432 ,436 ,'f'),
('FDF2P0'       ,436 ,440 ,'f'),
('FDF2P1'       ,440 ,444 ,'f'),
('FDF2LB'       ,444 ,448 ,'f'),
('FDF2OBS'      ,476 ,480 ,'f'),
('FDMCFLAG'     ,540 ,544 ,'i'),
('FDF2UNITS'    ,608 ,612 ,'f'),
('FDNOISE'      ,612 ,616 ,'f'),
('FDTEMPERATURE',628 ,632 ,'f'),
('FDPRESSURE'   ,632 ,636 ,'f'),
('FDRANK'       ,720 ,724 ,'f'),
('FDTAU'        ,796 ,800 ,'f'),
('FDF3FTSIZE'   ,800 ,804 ,'i'),
('FDF4FTSIZE'   ,804 ,808 ,'i'),
('FDF1OBS'      ,872 ,876 ,'f'),
('FDSPECNUM'    ,876 ,880 ,'i'),
('FDF2FTFLAG'   ,880 ,884 ,'i'),
('FDTRANSPOSED' ,884 ,888 ,'i'),
('FDF1FTFLAG'   ,888 ,892 ,'i'),
('FDF1SW'       ,916 ,920 ,'f'),
('FDF1UNITS'    ,936 ,940 ,'f'),
('FDF1LB'       ,972 ,976 ,'f'),
('FDF1P0'       ,980 ,984 ,'f'),
('FDF1P1'       ,984 ,988 ,'f'),
('FDMAX'        ,988 ,992 ,'f'),
('FDMIN'        ,992 ,996 ,'f'),
('FDF1ORIG'     ,996 ,1000,'f'),
('FDSCALEFLAG'  ,1000,1004,'i'),
('FDDISPMAX'    ,1004,1008,'f'),
('FDDISPMIN'    ,1008,1012,'f'),
('FDPTHRESH'    ,1012,1016,'f'),
('FDNTHRESH'    ,1016,1020,'f'),
('FD2DPHASE'    ,1024,1028,'f'),
('FDF2X1'       ,1028,1032,'i'),
('FDF2XN'       ,1032,1036,'i'),
('FDF1X1'       ,1036,1040,'i'),
('FDF1XN'       ,1040,1044,'i'),
('FDF3X1'       ,1044,1048,'i'),
('FDF3XN'       ,1048,1052,'i'),
('FDF4X1'       ,1052,1056,'i'),
('FDF4XN'       ,1056,1060,'i'),
('FDDOMINFO'    ,1064,1068,'f'),
('FDMETHINFO'   ,1068,1072,'f'),
('FDHOURS'      ,1132,1136,'f'),
('FDMINS'       ,1136,1140,'f'),
('FDSECS'       ,1140,1144,'f'),
('FDSRCNAME'    ,1144,1160,'s'),
('FDUSERNAME'   ,1160,1176,'s'),
('FDMONTH'      ,1176,1180,'f'),
('FDDAY'        ,1180,1184,'f'),
('FDYEAR'       ,1184,1188,'f'),
('FDTITLE'      ,1188,1248,'s'),
('FDCOMMENT'    ,1248,1408,'s'),
('FDLASTBLOCK'  ,1436,1440,'f'),
('FDCONTBLOCK'  ,1440,1444,'f'),
('FDBASEBLOCK'  ,1444,1448,'f'),
('FDPEAKBLOCK'  ,1448,1452,'f'),
('FDBMAPBLOCK'  ,1452,1456,'f'),
('FDHISTBLOCK'  ,1456,1460,'f'),
('FD1DBLOCK'    ,1460,1464,'f'),
('FDSCORE'      ,1480,1484,'f'),
('FDSCANS'      ,1484,1488,'f'),
('FDF3LB'       ,1488,1492,'f'),
('FDF4LB'       ,1492,1496,'f'),
('FDF2GB'       ,1496,1500,'f'),
('FDF1GB'       ,1500,1504,'f'),
('FDF3GB'       ,1504,1508,'f'),
('FDF4GB'       ,1508,1512,'f'),
('FDF2OBSMID'   ,1512,1516,'f'),
('FDF1OBSMID'   ,1516,1520,'f'),
('FDF3OBSMID'   ,1520,1524,'f'),
('FDF4OBSMID'   ,1524,1528,'f'),
('FDF2GOFF'     ,1528,1532,'f'),
('FDF1GOFF'     ,1532,1536,'f'),
('FDF3GOFF'     ,1536,1540,'f'),
('FDF4GOFF'     ,1540,1544,'f'),
('FDF2TDSIZE'   ,1544,1548,'i'),
('FDF1TDSIZE'   ,1548,1552,'i'),
('FDF3TDSIZE'   ,1552,1556,'i'),
('FDF4TDSIZE'   ,1556,1560,'i'),
('FD2DVIRGIN'   ,1596,1600,'f'),
('FDF3APODCODE' ,1600,1604,'f'),
('FDF3APODQ1'   ,1604,1608,'f'),
('FDF3APODQ2'   ,1608,1612,'f'),
('FDF3APODQ3'   ,1612,1616,'f'),
('FDF3C1'       ,1616,1620,'f'),
('FDF4APODCODE' ,1620,1624,'f'),
('FDF4APODQ1'   ,1624,1628,'f'),
('FDF4APODQ2'   ,1628,1632,'f'),
('FDF4APODQ3'   ,1632,1636,'f'),
('FDF4C1'       ,1636,1640,'f'),
('FDF2APODCODE' ,1652,1656,'f'),
('FDF1APODCODE' ,1656,1660,'f'),
('FDF2APODQ1'   ,1660,1664,'f'),
('FDF2APODQ2'   ,1664,1668,'f'),
('FDF2APODQ3'   ,1668,1672,'f'),
('FDF2C1'       ,1672,1676,'f'),
('FDF2APODDF'   ,1676,1680,'f'),
('FDF1APODQ1'   ,1680,1684,'f'),
('FDF1APODQ2'   ,1684,1688,'f'),
('FDF1APODQ3'   ,1688,1692,'f'),
('FDF1C1'       ,1692,1696,'f'),
('FDF1APOD'     ,1712,1716,'f'),
('FDF1ZF'       ,1748,1752,'i'),
('FDF3ZF'       ,1752,1756,'i'),
('FDF4ZF'       ,1756,1760,'i'),
('FDFILECOUNT'  ,1768,1772,'i'),
('FDSLICECOUNT' ,1772,1776,'i'),
('FDSLICECOUNT0',1772,1776,'i'),
('FDTHREADCOUNT',1776,1780,'i'),
('FDTHREADID'   ,1780,1784,'f'),
('FDSLICECOUNT1',1784,1788,'i'),
('FDCUBEFLAG'   ,1788,1792,'i'),
('FDOPERNAME'   ,1856,1888,'s'),
('FDF1AQSIGN'   ,1900,1904,'f'),
('FDF3AQSIGN'   ,1904,1908,'f'),
('FDF4AQSIGN'   ,1908,1912,'f'),
('FDF2OFFPPM'   ,1920,1924,'f'),
('FDF1OFFPPM'   ,1924,1928,'f'),
('FDF3OFFPPM'   ,1928,1932,'f'),
('FDF4OFFPPM'   ,1932,1936,'f')]





def load_peaks(fileName, reverse=False):
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
	peaks = []
	with open(fileName) as o:
		i=0
		for line in o:
			splt = line.split()
			try:
				pos = np.array(splt[-2:], dtype=float)
				if reverse:
					pos = pos[::-1]
				if len(splt)==3 and len(pos)==2:
					peaks.append([splt[0], pos])
				elif len(pos)==2:
					peaks.append([i, pos])
					i += 1
				else:
					raise ValueError
			except ValueError:
				print("Line ignored in peaks file: {}".format(repr(line)))
		cm = plt.get_cmap('brg', len(peaks))
		for i, peak in enumerate(peaks):
			peak += [cm(i)]
	return peaks


def load_ccpnpeaks(fileName):
	peaks = []
	with open(fileName) as o:
		hdr = next(o)
		i=0
		for line in o:
			splt = line.split()
			pos = np.array(splt[2:4], dtype=float)
			peaks.append([splt[4], pos])

		cm = plt.get_cmap('brg', len(peaks))
		for i, peak in enumerate(peaks):
			peak += [cm(i)]
	return peaks




class Axis(object):
	"""
	Stores information about the axis ppm values.
	Can provide unit conversion between points, ppm and Hz values.
	Note all scales are returned in descending order
	"""
	def __init__(self, points, carrier, spectralWidth, observedFrequency,
		dimension, label):
		"""
		Returns Axis object to store ppm axis info.

		Parameters
		----------
		points : int
			number of points along the axis
		carrier : float
			centre frequency of the axis in Hz
		spectralWidth : float
			spectral width of the axis in Hz
		observedFrequency : float
			acquired NMR spectrometer frequency in MHz
		dimension : int
			dimension of the current axis
		label : str
			nucleus identifier e.g. 1H, 15N, 13C

		Returns
		-------
		Axis : object
			stores axis information and provides unit conversion between
			points, ppm and Hz scales
		"""
		self.p   = points
		self.car = carrier
		self.sw  = spectralWidth
		self.obs = observedFrequency
		self.dim = dimension
		self.lbl = label

	def __str__(self):
		return "<Axis {0}:{1:>3}>".format(self.dim, self.lbl)

	def __repr__(self):
		return self.__str__()

	@property
	def hz_scale(self):
		"""
		Returns
		-------
		array : numpy.ndarray of floats
			the axis values in Hz in descending order
		"""
		return np.linspace(self.car+self.sw/2, self.car-self.sw/2, self.p)

	@property
	def ppm_scale(self):
		"""
		Returns
		-------
		array : numpy.ndarray of floats
			the axis values in ppm in descending order
		"""
		return self.hz_scale / self.obs

	@property
	def ppm_limits(self):
		"""
		Returns
		-------
		tuple : (float, float)
			the high and low ppm limits respectively
		"""
		scale = self.ppm_scale
		return scale.max(), scale.min()

	def f(self, ppm):
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
		hzpp = self.sw / float(self.p-1)
		loc = (-ppm * self.obs + self.car + self.sw/2) / hzpp
		return loc

	def i(self, ppm):
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
		return np.argmin(np.abs(self.ppm_scale - ppm))

	def __getitem__(self, slic):
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
			return slice(self.i(start), self.i(stop)+1)
		else:
			return self.i(slic)

	def new(self, slic):
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
			p = len(scale)
			sw = np.ptp(scale)
			car = (scale.min() + scale.max()) / 2.0
			return self.__class__(p, car, sw, self.obs, self.dim, self.lbl)
		else:
			loc = self.f(slic)
			low = int(loc)
			high = low + 1
			weight = loc - low
			return low, high, weight



class Spectrum(object):
	"""
	Stores spectrum data matrix and axes info.
	"""
	@staticmethod
	def reorder_submatrix(data, shape, submatrix_shape):
		"""
		Copied from nmrglue version 0.6
		Reorder processed binary Bruker data.

		Parameters
		----------
		data : array

		shape : tuple
			Shape of final data.
		submatrix_shape : tuple
			Shape of submatrix.

		Returns
		-------
		rdata : array
			Array in which data has been reordered and correctly shaped.

		"""
		if submatrix_shape is None or shape is None:
			return data

		# do nothing to 1D data
		if len(submatrix_shape) == 1 or len(shape) == 1:
			return data

		rdata = np.empty(shape, dtype=data.dtype)
		sub_per_dim = [int(i / j) for i, j in zip(shape, submatrix_shape)]
		nsubs = np.product(sub_per_dim)
		data = data.reshape([nsubs] + list(submatrix_shape))

		for sub_num, sub_idx in enumerate(np.ndindex(tuple(sub_per_dim))):
			sub_slices = [slice(i * j, (i + 1) * j) for i, j in
						  zip(sub_idx, submatrix_shape)]
			rdata[tuple(sub_slices)] = data[sub_num]
		return rdata

	@staticmethod
	def read_procs(fileName):
		lineMatch = re.compile("\#\#\$(.*)=[\s+]?(.*)")
		valueMatch = re.compile("<(.*)>")
		dic = {}
		with open(fileName) as o:
			for line in o:
				m = lineMatch.match(line)
				if m:
					key, value = m.groups()
					v = valueMatch.match(value)
					if v:
						dic[key] = v.group(1)
					else:
						try:
							f = float(value)
							i = int(f)
							if f==i:
								dic[key] = i
							else:
								dic[key] = f
						except ValueError:
							dic[key] = value    
		return dic

	@staticmethod
	def read_spectrum(fileName):
		with open(fileName, 'rb') as o:
			data = np.frombuffer(o.read(), dtype='<i4')
		data = np.asarray(data, dtype=float)
		return data

	@staticmethod
	def read_clevels(fileName):
		with open(fileName) as o:
			s = o.read().replace('\n',' ')
		lvls = re.search(r'\#\#\$LEVELS=\s?\(.*\)(.*?)\#\#\$', s).group(1)
		return np.trim_zeros(np.array(lvls.split(), dtype=float))


	@classmethod
	def load_bruker(cls, spectrumDir):
		# Fetch directories
		procs = {}
		clevels = None
		for fileName in os.listdir(spectrumDir):
			fullDir = os.path.join(spectrumDir, fileName)

			if fileName in ('1r', '2rr', '3rrr', '4rrrr'):
				specFile = fullDir

			elif re.search("proc[0-9]?s", fileName):
				num = re.search("proc([0-9])?s", fileName).group(1)
				if not num:
					dim = 1
				else:
					dim = int(num)
				procs[int(dim)] = cls.read_procs(fullDir)

			elif re.search("clevels", fileName):
				clevels = fullDir

		# Get dimensions
		axes = []
		actualShape = []
		subMatrixShape = []
		for dim in sorted(procs):
			proc = procs[dim]
			p   = proc['SI']
			car = proc['OFFSET']*proc['SF'] - 0.5*proc['SW_p']
			sw  = proc['SW_p']
			obs = proc['SF']
			lbl = proc['AXNUC']
			axes.append(Axis(p, car, sw, obs, dim, lbl))
			actualShape.append(p)
			subMatrixShape.append(proc.get('XDIM', p))

		scale = 2.0**(-procs[1].get('NC_proc'))
		axes = sorted(axes, key=lambda x: -x.dim)

		# Read spectrum data
		data = cls.read_spectrum(specFile)
		data = cls.reorder_submatrix(data, actualShape[::-1], 
			subMatrixShape[::-1]) / scale

		# Read contour levels
		if clevels:
			clevels = cls.read_clevels(clevels)

		return cls(data, axes, clevels)

	@classmethod
	def load_pipe(cls, spectrum):
		with open(spectrum, 'rb') as buff:
			rawHdr = buff.read(2048)
			data = np.fromfile(buff, 'float32')

		hdr = {}
		for label, start, end, dtype in pipe_hdr_key:
			rawBytes = rawHdr[start:end]
			num = end - start
			if dtype in ('f', 'i'):
				formBytes = "{}{}".format((end-start)//4, 'f')
			else:
				formBytes = "{}{}".format(end-start, dtype)
			value = struct.unpack(formBytes, rawBytes)[0]
			if dtype=='s':
				value = value.strip(b"\x00").decode()
			if dtype=='i':
				value = int(value)
			hdr[label] = value

		axes = []
		dims = []
		for i in range(1,4):
			dim = hdr['FDDIMORDER{}'.format(i)]
			dims.append(dim-1)
			x1 = hdr['FDF{}X1'.format(dim)]
			xn = hdr['FDF{}XN'.format(dim)]
			if xn == 0:
				xn = hdr['FDF{}FTSIZE'.format(dim)] - 1
			p = xn - x1	+ 1
			sw  = hdr['FDF{}SW'.format(dim)]
			car = hdr['FDF{}ORIG'.format(dim)] + sw/2.0
			obs = hdr['FDF{}OBS'.format(dim)]
			lbl = hdr['FDF{}LABEL'.format(dim)]
			new_axis = Axis(p, car, sw, obs, i, lbl)
			axes.append(new_axis)

		axes = sorted(axes, key=lambda x: -x.dim)
		data = data.reshape(*[axis.p for axis in axes])
		data /= np.std(data)
		clevels = cls.make_contours(5, 20, 10)
		spec = cls(data, axes, clevels)
		# spec.reorder_axes([1,0,2])
		return spec



	@staticmethod	
	def make_contours(lowest, highest, number):
		pos = lowest+(highest-lowest)*(np.arange(0,number)/float(number-1))**(2.**0.5)
		neg = -pos[::-1]
		return np.concatenate([neg, pos])


	def __init__(self, data, axes, contours=None):
		self.data = data
		self.axes = axes
		self.cont = contours

	def __getitem__(self, slices):
		new_slice = []
		new_axes = []
		for axis, slic in zip(self.axes, slices):
			new_slice.append(axis[slic])
			new_axis = axis.new(slic)
			if isinstance(new_axis, Axis):
				new_axes.append(new_axis)
		return self.__class__(self.data[tuple(new_slice)], new_axes, self.cont)
		
	@property
	def extent(self):
		ex = ()
		for axis in self.axes:
			ex += axis.ppm_limits[::-1]
		return ex[::-1]

	@property
	def poscont(self):
		tmp = self.cont[np.where(self.cont>0)]
		if len(tmp)>0:
			return tmp
		else:
			return None

	@property
	def negcont(self):
		tmp = self.cont[np.where(self.cont<0)]
		if len(tmp)>0:
			return tmp
		else:
			return None

	def projection(self, axis):
		data = np.max(self.data, axis=axis)
		axes = [ax for i,ax in enumerate(self.axes) if i!=axis]
		return self.__class__(data, axes, self.cont)

	def reorder_axes(self, newAxisOrder):
		self.axes = [self.axes[i] for i in newAxisOrder]
		for i, axis in enumerate(self.axes):
			axis.dim = 3-i
		axis_nums = [i for i, axis in enumerate(self.axes)]
		self.data = np.moveaxis(self.data, axis_nums, newAxisOrder)

	def transpose(self):
		new_order = list(range(len(self.axes)))[::-1]
		self.reorder_axes(new_order)



def print_progress(counter, total):
	sys.stdout.write("\rProgress: {:7.1f}%".format((100.*counter)/total))
	sys.stdout.flush()
	if counter==total:
		print("")


def even_divide(lst, n):
	p = len(lst) // n
	if len(lst)-p > 0:
		return [lst[:p]] + even_divide(lst[p:], n-1)
	else:
		return [lst]

axis_dict = {
	'z':0,
	'y':1,
	'x':2
}



if args:
	width = args.width
	if args.ccpnpeaks:
		peaks = load_ccpnpeaks(args.peaks)
	else:
		peaks = load_peaks(args.peaks, reverse=args.opposite)

	colours = [('b','g'),('r','m'),('k','o')]
	projAxis = axis_dict[args.projectionaxis]
	axisOrder = [axis_dict[i] for i in args.axisorder]
	print(axisOrder)

	if args.pages:
		numfigs = args.pages
		figs = [plt.figure(figsize=(16.5,11.7)) for i in range(numfigs)]

	else:
		numfigs = 1
		figs = [plt.figure(figsize=(2.7*width*len(peaks),11))]

	print("Making subplots ...")
	total = len(peaks)
	progress = 0
	plotAxes = []
	for peakset, fig in zip(even_divide(peaks, numfigs), figs):
		hide_axis = False
		subpltcnt = 1
		for lbl, peak, lblcol in peakset:
			progress += 1
			print_progress(progress, total)

			ax = fig.add_subplot(1, len(peakset), subpltcnt)
			if subpltcnt==1:
				setattr(ax, 'customStartingAxisBool', True)
			else:
				setattr(ax, 'customStartingAxisBool', False)
			plotAxes.append(ax)
			subpltcnt += 1
			setattr(ax, 'customPeakPosition', peak)
			c3p, c2p = peak
			ax.text(.5,.94,"{:3.1f}".format(c2p),horizontalalignment='center',
				transform=ax.transAxes, rotation=90, backgroundcolor='1')
			ax.set_title(str(lbl), color=lblcol, rotation=90, 
				verticalalignment='bottom')

	print("Plotting strips ...")
	total = len(plotAxes)*len(args.dataset)
	progress = 0
	for dataset, col in zip(args.dataset, colours):
		try:
			spec = Spectrum.load_bruker(dataset)
		except:
			spec = Spectrum.load_pipe(dataset)

		spec.reorder_axes(axisOrder)

		if args.contours:
			spec.cont = spec.make_contours(*args.contours)

		if args.range is not None:
			h1p, l1p = args.range
		else:
			h1p, l1p = spec.axes[0].ppm_limits

		for ax in plotAxes:
			progress += 1
			print_progress(progress, total)

			c3p, c2p = ax.customPeakPosition # F3 F2 peak position
			h3p, l3p = c3p+width*0.5, c3p-width*0.5 # F3 high and low
			strip = spec[h1p:l1p,c2p,h3p:l3p]

			if spec.poscont is not None:
				ax.contour(strip.data, strip.poscont, extent=strip.extent, 
					colors=col[0], linewidths=0.05)
			if spec.negcont is not None:
				ax.contour(-strip.data, -strip.negcont[::-1], extent=strip.extent, 
					colors=col[1], linewidths=0.05)

	print("Adjusting axes ...")
	total = len(plotAxes)
	progress = 0
	for ax in plotAxes:
		progress += 1
		print_progress(progress, total)
		c3p, c2p = ax.customPeakPosition
		
		ax.set_xticks([c3p])
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax.invert_xaxis()

		if args.range is not None:
			h1p, l1p = args.range
		else:
			h1p, l1p = ax.get_ylim()
		ax.set_ylim(h1p, l1p)
		ax.set_yticks(np.linspace(int(l1p),int(h1p)+1,40))
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		ax.yaxis.grid(linestyle='dotted')
		ax.invert_yaxis()

		if ax.customStartingAxisBool:
			ax.tick_params(right='off')			
		else:
			ax.yaxis.set_ticklabels([])
			ax.yaxis.set_ticks_position('none')

	print('Drawing figures ...')
	total = len(figs)
	progress = 0
	fileName = 'strips.pdf'
	with PdfPages(fileName) as pdf:
		for fig in figs:
			fig.subplots_adjust(wspace=0)
			fig.autofmt_xdate(rotation=90, ha='center')
			pdf.savefig(fig, bbox_inches='tight')
			progress += 1
			print_progress(progress, total)
	print("{} file written".format(fileName))

	if args.hsqc:
		print("Plotting HSQC ...")
		hsqc = Spectrum.load_bruker(args.hsqc)
	else:
		print("Plotting projection ...")
		hsqc = spec.projection(projAxis)

	fig = plt.figure(figsize=(16.5,11.7))
	ax = fig.add_subplot(111)

	if spec.poscont is not None:
		ax.contour(hsqc.data, hsqc.poscont, colors='b', 
			extent=hsqc.extent, linewidths=0.05)
	if spec.negcont is not None:
		ax.contour(hsqc.data, hsqc.negcont, colors='g', 
			extent=hsqc.extent, linewidths=0.05)
	for lbl, peak, lblcol in peaks:
		ax.plot(*peak, color='r', marker='x')
		ax.annotate(lbl, xy=peak, color=lblcol, fontsize=5)

	ax.invert_xaxis()
	ax.invert_yaxis()
	hsqcFileName = 'hsqc.pdf'
	fig.savefig(hsqcFileName, bbox_inches='tight')
	print("{} file written".format(hsqcFileName))



# def timer(func):
# 	import time
# 	def wrapper(*args, **kwargs):
# 		start = time.time()
# 		output = func(*args, **kwargs)
# 		finish = time.time()
# 		print("Calculation time: {:.2g} sec".format(finish-start))
# 		return output
# 	return wrapper

