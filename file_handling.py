'''
This script contains all file handling functions.

PB 2025
'''

# Import all necessary modules
import numpy as np
import sys
# sys.path.append('/root/Research_Project/your/your/your.py')

import your
from your import Your
from astropy.io import fits
# from your.candidate import Candidate
# from your.utils.plotter import plot_h5
import os
import struct
import mmap
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import tempfile
import subprocess
from astropy.coordinates import Angle

################################
#   READ DATA FROM FILTERBANK  #
################################

def get_data(yourobj, nstart, nsamp=None, npoln=1, keys='infer'):
	'''
	Function to read out data from filterbank file and returns a dictionary with the measured data in all polarizations
	Inputs:
		yourobj (your object)	: 	Your object that contains metadata of a given fits or file file
		nstart (int)		:	Starting sample of the data that is wanted (default=0)
		nsamp (None/int)	:	Number of samples to return (default=None -> all samples)
		npoln (int)		: 	Number of polarisations. Should be 1 (intensity) or 4 (full pol or coherence) (default=1)
		keys (list)		:	String or list with strings containing the keys (polarization names) to be returned (default='infer')

	Returns:
		data (2D array)	:	Array containing data of one polarization, if npoln==1
		data_dict (dictionary)	:	Dictionary with four keys, each pointing to a 2D array per polarization
	'''

	assert npoln == 1 or npoln == 4

	if keys == 'infer' and npoln == 4:
		keys = ['LL', 'RR', 'LR', 'RL']

	if nsamp == None:
		nsamp = int(yourobj.native_nspectra)

	# Reading in the data
	data = yourobj.get_data(nstart=nstart, nsamp=nsamp, npoln=npoln).T

	if npoln == 1:
		return data

	res = {}
	# Store the data in dict
	for i in range(npoln):
		res[keys[i]] = data[:, i, :]

	return res


################################
#          SORT SOURCES        #
################################

# Observation targets
targets = ['ADLEO', 'PSV1', 'ILTJ1101', 'wxuma', 'eqpeg', 'WXUMA', 'EQPEG']
# , 'CASA', 'CYGA']


def assign_calibration_by_order(files, file_path):
	'''
	Assigns calibration pulsars to sources based on their order in the observation list.
	Inputs:
		files (list, str)	: List of all the observation files with path
		file_path (str)	: String containing the directory of the file
	Returns:
		calibration_map (dict) : Mapping of each source to its assigned calibration pulsar
		source_file (list, str): List of source observations file path and name
		pulsar_file (list, str): List of calibration pulsar file path and name
		source_counts (dict)   : Dictionary mapping each soruce to its totla observation count
	'''

	pulsar_list = []
	pulsar_file = []
	source_list = []
	source_file = []
	source_counts = {}

	# Extract the source names from the files
	for file in files:
		fname = '{}{}'.format(file_path, file)
		try:
			your_object = your.Your(fname)
		except struct.error as e:
			print('Skipping file {} due to error: {}'.format(fname, e))
			continue

		hdr = your_object.your_header
		source = hdr.source_name

		# Count observations per source
		source_counts[source] = source_counts.get(source, 0) + 1

		# sort the files and avoid doubling for consecutive observations
		if len(source_list)==0:
			source_list.append(source)

		elif source_list[-1] != source:
			source_list.append(source)

		if source not in targets:
			pulsar_file.append(fname)
			pulsar_list.append(source)
		else:
			source_file.append(fname)


	calibration_map = {}
	current_pulsar = None
	max_sources = 2			# How many sources are to be calibrated by one pulsar measurement

	source_count = 0

	for i, source in enumerate(source_list):
		if source in pulsar_list:
			# Update current pulsar and reset source count
			current_pulsar = source
			source_count = 0

		else:
			# Assign the current pulsar if there are less than max_source already assigned
			if source_count < max_sources:
				calibration_map[source] = current_pulsar
				source_count += 1
			else:
				# If we reached max_sources for this pulsar, assign to the next pulsar
				if i +1 < len(source_list) and source_list[i+1] in pulsar_list:
					# If the next source is a pulsar, use it
					calibration_map[source] = source_list[i+1]
					source_count += 1
				else:
					# Otherwise, keep using the last pulsar
					calibration_map[source] = current_pulsar
					source_count += 1

	return calibration_map, source_file, pulsar_file, source_counts
	
	
	
################################
#       SAVE DATA AS FITS      #
################################
	
def save_to_fits(data, hdr, obs_name, fname, polarization, counter, tsamp):
	'''
	Save a 2D array of data as a .fits file

	Inputs:
		data (2D array): 	The 2D array containing the data to be saved
		hdr (your)     :	Your header of the dataset
		obs_name (str) : 	Name of the observation
		fname (str)    :	Name of the underlying filterbank file
		polarization(str):	Measured polarization
		counter (int)  :	If more than one, indicates number of the file
		tsamp (float)  :	How many seconds pass per sample
	Returns:
		None, saves a .fits file into a given directory
	'''

	path = '/root/Research_Project/addpathhere/'
	your_hdr = Your(fname).your_header
	source_name = hdr.source_name

	file_name = '{}_{}_{}_{}.fits'.format(obs_name, source_name, polarization, counter)
	file_path = '{}{}'.format(path, file_name)

	# Ensure the input si a 2D numpy array
	if not isinstance(data, np.ndarray) or data.ndim !=2:
		raise ValueError("Input data must be a 2D numpy array.")

	# Create a FITS header
	hdr = fits.Header()
	hdr['SIMPLE']   = (True, "Conforms to FITS standard")
	hdr['BITPIX']   = 32  # 32-bit floating point data
	hdr['NAXIS']    = 2  # 2D data
	hdr['NAXIS1']   = data.shape[1]  # Columns (time samples)
	hdr['NAXIS2']   = data.shape[0]  # Rows (frequency channels)

	# Standard keywords
	hdr['OBSERVAT'] = ('WSRT', "Observatory")  # Name of the observatory
	hdr['TELESCOP'] = ('RT1', "Telescope")  # Telescope name
	hdr['OBSERVER'] = ('Paul Baecke', "Observer")  # Observer's name
	hdr['DATE-OBS'] = (str(your_hdr.tstart_utc), "Observation start (UTC)")  # Observation date

	# Frequency and sampling information
	hdr['FCH1']     = (your_hdr.fch1, "First channel frequency (MHz)")  # Placeholder frequency
	hdr['FOFF']	= (your_hdr.foff, "Channel bandwidth (MHz)")
	hdr['BW']	= (your_hdr.bw, "Total bandwidth (MHz)")
	hdr['TSAMP']    = (tsamp, "Sampling time (s)")  # Placeholder sampling time

	# Additional custom keywords
	hdr['SRCNAME']	= (source_name, "Source name")
	hdr['RADEG']	= (your_hdr.ra_deg, "RsA (deg)")
	hdr['DECDEG']	= (your_hdr.dec_deg, "DEC (deg)")

	# Information on the data dimensions
	hdr['NSPEC']	= (your_hdr.nspectra, "Number of spectra")
	hdr['NPOL']	= (1, "Number of polarizations")

	print(hdr)
	# Create a Fits PrimaryHDU object from the data array
	hdu = fits.PrimaryHDU(data, header = hdr)

	# Create a HDUList object containing the PrimaryHDU
	# hdulist = fits.HDUList([hdu])

	# Write the HDUList to the specified file
	hdu.writeto(file_path, overwrite=True)

	print('{} Data saved successfully to {}'.format(source_name, file_path))


################################
#   READ DATA FROM FITS FILE   #
################################

def load_from_fits(file_path):
	'''
	Load a 2D array of data as a .fits file

	Inputs: 
		file_path (str):	String containing the path to the fits file
	Returns:
		data (2D array):	The 2D numpy array of the selected data
	'''

	# open the fits file
	with fits.open(file_path) as hdul:
		# Extract the data from the primary HDU
		data = hdul[0].data

	# Ensure the input is a 2D numpy array
	if not isinstance(data, np.ndarray) or data.ndim !=2:
		raise ValueError("Input data must be a 2D numpy array.")

	print('Data successfully loaded.')

	return data


################################
#    SAVE DATA AS FILTERBANK   #
################################


def save_to_fil(input_data, old_hdr, obs_name, stokes, nchans, start_channel, tsamp, path, fname):
	'''
	This function writes a filterbank file by first creating a temporary raw filterbank file and then executing a terminal command to call your_writer.py with the correct options.

	Inputs:
		data (2D array) 	: Data to be saved. For a single polarization, the shape should be (nsamps, nchans)
		oldhdr (your_header)	: Header of the original filterbank file
		obs_name (str)		: Observation code
		stokes (str)		: "I" or "V"
		nchan  (int)		: Number of frequency channels
		start_channel (int)	: Index of the first frequency 
		tsamp (float)		: seconds that pass per sample
		path (str)		: Directory where to save the filterbank file
		fname (str)		: File name
	Outputs:
		None			: Writes filterbank file to disk
	'''


	# Defining the filterbank file name
	# fpath = "/home/paulbaecke/Research_Project/ao_data/ao_test/spl/"
	# fname = "{}_{}_wb_aobp_{}_{}".format(obs_name, old_hdr.source_name, stokes, number)

	# Full file path of the new filterbank file
	fdir = "{}/{}.fil".format(path, fname)
	
	new_fch1 = old_hdr.fch1 + start_channel * old_hdr.foff
	
	# Quick function to transfor the source coordinates
	format_coord = lambda deg, unit: float(Angle(deg, unit='deg').to_string(unit=unit, sep='', precision=2, alwayssign=False, pad=True))

	# Creating the header of the new filterbank file
	sigproc_object = your.formats.filwriter.make_sigproc_object(
		rawdatafile = fname, 
		source_name = old_hdr.source_name,
		nchans = nchans,
		foff = old_hdr.foff,
		fch1 = new_fch1, 
		tsamp = tsamp, 
		tstart = old_hdr.tstart,
		src_raj = format_coord(old_hdr.ra_deg, 'hourangle'),
		src_dej = format_coord(old_hdr.dec_deg, 'deg'),
		machine_id = 0,
		nbeams = 0, 
		ibeam = 0, 
		nbits = old_hdr.nbits,
		nifs = len(stokes), 
		barycentric = 0,
		pulsarcentric = 0,
		telescope_id = 2,
		data_type = 0,
		az_start = -1, 
		za_start = -1,
	)
	
	# Write the header in the files
	sigproc_object.write_header(fdir)

	data_to_write = input_data.T	
	
	# Write the data to the new file
	sigproc_object.append_spectra(data_to_write, fdir)

	print('Filterbank file saved: {}.fil\n{}'.format(fname, fdir))


