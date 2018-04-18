import numpy as np
import os
import sys
import re

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
		help="directory of nmrPipe 3D data",type=str,nargs='+')
	parser.add_argument('-p','--peaks',
		help="2 dimensional peak list with 'f3 f2' in ppm separated by spaces",
		type=str)
	parser.add_argument('-w','--width',
		help="strip width in ppm, optional: default=0.15",
		type=float, default=0.15)
	parser.add_argument('-c','--contours',
		help="3 numbers specifying the contour levels: 'low high number'",
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
	args = parser.parse_args()
else:
	args = None



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
				if len(splt)==3:
					peaks.append([splt[0], pos])
				else:
					peaks.append([i, pos])
					i += 1
			except ValueError:
				print("Line ignored in peaks file: {}".format(repr(line)))
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



a = Axis(50, 0.0, 800, 800, 1, '1H')


print(a.new(0.2))

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
			rdata[sub_slices] = data[sub_num]
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
		lvls = re.search('\#\#\$LEVELS=\s?\(.*\)(.*?)\#\#\$', s).group(1)
		return np.trim_zeros(np.array(lvls.split(), dtype=float))


	@classmethod
	def load_bruker(cls, spectrumDir):
		# Fetch directories
		procs = {}
		clevels = None
		for fileName in os.listdir(spectrumDir):
			fullDir = os.path.join(spectrumDir, fileName)
			if re.search("[0-9][r]+[^i]", fileName):
				specFile = fullDir

			elif re.search("proc[0-9]?s", fileName):
				dim = filter(str.isdigit, fileName)
				if not dim:
					dim = 1
				procs[int(dim)] = cls.read_procs(fullDir)

			elif re.search("clevels", fileName):
				clevels = fullDir

		# Get dimensions
		axes = []
		actualShape = []
		subMatrixShape = []
		for dim, proc in procs.items():
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
	def load_pipe(cls, spectrumDir):
		import nmrglue as ng
		from pprint import pprint
		dic, data = ng.pipe.read(spectrumDir)
		axes = []
		for i in range(1,4):
			dim = int(3-dic['FDDIMORDER{}'.format(i)])
			p = data.shape[dim]
			sw  = dic['FDF{}SW'.format(i)]
			car = dic['FDF{}ORIG'.format(i)] + sw/2.0
			obs = dic['FDF{}OBS'.format(i)]
			lbl = dic['FDF{}LABEL'.format(i)]
			new_axis = Axis(p, car, sw, obs, dim, lbl)
			axes.append(new_axis)
		axes = sorted(axes, key=lambda x: x.dim)
		std = np.std(data)
		cont = cls.make_contours(8*std, 50*std, 10)
		cont = np.array(list(-cont[::-1])+list(cont))

		return cls(data, axes, cont)

	@staticmethod	
	def make_contours(lowest, highest, number):
		return lowest+(highest-lowest)*(np.arange(0,number)/float(number-1))**(2.**0.5)


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

		return self.__class__(self.data[new_slice], new_axes, self.cont)
		
	@property
	def extent(self):
		ex = ()
		for axis in self.axes:
			ex += axis.ppm_limits[::-1]
		return ex[::-1]

	@property
	def poscont(self):
		return self.cont[np.where(self.cont>0)]

	@property
	def negcont(self):
		return self.cont[np.where(self.cont<0)]

	def projection(self, axis):
		data = np.max(self.data, axis=axis)
		axes = [ax for i,ax in enumerate(self.axes) if i!=axis]
		return self.__class__(data, axes, self.cont)

	def reorder_axes(self, newAxisOrder):
		self.axes = [self.axes[i] for i in newAxisOrder]
		self.data = np.moveaxis(self.data, [0,1,2], newAxisOrder)


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
	peaks = load_peaks(args.peaks, reverse=args.opposite)
	
	colours = [('b','g'),('r','m'),('k','o')]
	projAxis = axis_dict[args.projectionaxis]
	axisOrder = [axis_dict[i] for i in args.axisorder]

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
			ax.text(.5,.97,"{:3.1f}".format(c2p),horizontalalignment='center',
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

			c3p, c2p = ax.customPeakPosition
			h3p, l3p = c3p+width*0.5, c3p-width*0.5
			strip = spec[h1p:l1p,c2p,h3p:l3p]

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				ax.contour(strip.data, strip.poscont, extent=strip.extent, 
					colors=col[0], linewidths=0.05)
				ax.contour(strip.data, strip.negcont, extent=strip.extent, 
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

	ax.contour(hsqc.data, hsqc.poscont, colors='b', 
		extent=hsqc.extent, linewidths=0.05)
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





