import nmrglue as ng
import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import argparse
import warnings

from pprint import pprint


long_description = """
Make a strip plot from a 3D Bruker data set and 2D peak list. 
Typical usage: 
python strippy.py -d 14/pdata/3 -p peaks.list -w 0.15 -c 0.1 10 10 -r 0.5 6.0. 
This would plot processed data 3 from dataset 14.
The peak list has columns 'f3 f2' separated by spaces.
The strips will have with 0.15 ppm.
The contour levels are plotted from 0.1 to 10 with 10 levels total,
each contour has i**(2**0.5) scaling, and spectrum values are normalised by the
standard deviation.
Plotting in the f1 dimension in from 0.5 to 6.0 ppm.
"""


if __name__=='__main__':
	parser = argparse.ArgumentParser(description=long_description)
	parser.add_argument('-d','--dataset',
		help="directory of nmrPipe 3D data",type=str)
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
		help="Range to be plotted in f1 dimension e.g. '0.0 6.0'",
		type=float, nargs=2)
	parser.add_argument('-s','--hsqc',
		help="directory of nmrPipe 2D HSQC data",
		type=str)
	args = parser.parse_args()
else:
	args = None



def load_peaks(fileName):
	peaks = []
	with open(fileName) as o:
		for line in o:
			try:
				pos = np.array(line.split()[-2:], dtype=float)
				peaks.append(pos)
			except ValueError:
				print("Line ignored in peaks file: {}".format(repr(line)))
	return peaks


def load_spectrum_bruker(brukerProcDir):
	dic, data = ng.bruker.read_pdata(brukerProcDir)
	udic = ng.bruker.guess_udic(dic, data)
	proc = dic['procs']
	proc2 = dic['proc2s']
	proc3 = dic['proc3s']
	udic[2]['car'] = proc['OFFSET']*proc['SF'] - 0.5*proc['SW_p']
	udic[2]['sw'] = proc['SW_p']
	udic[1]['car'] = proc2['OFFSET']*proc2['SF'] - 0.5*proc2['SW_p']
	udic[1]['sw'] = proc2['SW_p']
	udic[0]['car'] = proc3['OFFSET']*proc3['SF'] - 0.5*proc3['SW_p']
	udic[0]['sw'] = proc3['SW_p']
	scales = {i:ng.fileiobase.uc_from_udic(udic, dim=i) for i in range(udic['ndim'])}
	return data/np.abs(data).mean(), scales


def load_spectrum_pipe(pipeDir):
	dic, data = ng.pipe.read(pipeDir)
	ndim = len(data.shape)
	scales = {i:ng.fileio.pipe.make_uc(dic, data, dim=i) for i in range(ndim)}
	return data/np.abs(data).mean(), scales


def load_spectrum(spectrumDir):
	try:
		a, b = load_spectrum_bruker(spectrumDir)
	except:
		pass

	try:
		a, b = load_spectrum_pipe(spectrumDir)
	except:
		pass

	return a, b



def make_contours(lowest, highest, number):
	return lowest+(highest-lowest)*(np.arange(0,number)/float(number-1))**(2.**0.5)



if args:
	width = args.width
	peaks = load_peaks(args.peaks)
	cm = plt.get_cmap('brg', len(peaks))
	print("Loading data ...")
	data, scales = load_spectrum(args.dataset)
	s1, s2, s3 = scales[0], scales[1], scales[2]

	if args.contours is None:
		std = np.std(data)
		cont = make_contours(5*std,50*std,10)
	else:
		cont = make_contours(*args.contours)

	if args.range is None:
		h1p, l1p = s1.ppm_limits()
		h1i, l1i = 0, len(s1.ppm_scale())
	else:
		l1p, h1p = args.range
		maxi, mini = s1.ppm_limits()
		if l1p<mini:
			l1p = mini
		if h1p>maxi:
			h1p = maxi
		h1i, l1i = (s1.i(i, unit='ppm') for i in (h1p, l1p))


	print("Plotting strips ...")
	fig = plt.figure(figsize=(2.7*width*len(peaks),11))
	fig.subplots_adjust(wspace=0)

	hide_axis = False

	for i, peak in enumerate(peaks):
		ax = fig.add_subplot(1, len(peaks), i+1)

		c3p, c2p = peak
		
		h3p, l3p = c3p+width*0.5, c3p-width*0.5
		h3i, l3i = (s3.i(i, unit='ppm') for i in (h3p, l3p))

		c2f = s2.f(c2p, unit='ppm')
		h2i, l2i = int(c2f+1), int(c2f)

		striph = (c2f - l2i)*data[h1i:l1i,h2i,h3i:l3i]
		stripl = (h2i - c2f)*data[h1i:l1i,l2i,h3i:l3i]
		strip = (stripl + striph)*0.5

		lims = (h3p,l3p,h1p,l1p)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			ax.contour( strip, cont, extent=lims, colors='k', linewidths=0.2)
			ax.contour(-strip, cont, extent=lims, colors='r', linewidths=0.2)
		ax.invert_xaxis()
		ax.invert_yaxis()
		ax.text(.5,.97,"{:3.1f}".format(c2p),horizontalalignment='center',
			transform=ax.transAxes, rotation=90, backgroundcolor='1')

		ax.set_xticks([c3p])
		ax.set_yticks(np.linspace(int(l1p),h1p,40))
		ax.yaxis.grid(linestyle='dotted')
		if hide_axis:
			ax.yaxis.set_ticklabels([])
			ax.yaxis.set_ticks_position('none')
		hide_axis = True

		ax.set_title(str(i), color=cm(i))
		
	
	fig.autofmt_xdate(rotation=90, ha='center')
	fileName = 'strips.pdf'
	fig.savefig(fileName, bbox_inches='tight')
	print("{} file written".format(fileName))


	if args.hsqc:
		data, scales = load_spectrum(args.hsqc)
		s2, s3 = scales[0], scales[1]
	else:
		data = np.ptp(data, axis=0)

	std = np.std(data)
	cont = make_contours(5*std,30*std,10)
	
	fig = plt.figure(figsize=(11,8))
	ax = fig.add_subplot(111)

	ax.contour(data, cont, colors='k', extent=s3.ppm_limits()+s2.ppm_limits(),
		linewidths=0.2)
	for i, peak in enumerate(peaks):
		ax.plot(*peak, color='r', marker='x')
		ax.annotate(str(i), xy=peak, color=cm(i))

	ax.invert_xaxis()
	ax.invert_yaxis()
	# ax.set_xlim(11, 5.5)
	# ax.set_ylim(136,100)
	fig.savefig('hsqc.pdf')





















