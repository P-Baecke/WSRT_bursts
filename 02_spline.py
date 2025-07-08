'''
This script is used to 
1) read in double flagged filterbank files
2) Plot (and save) the dynamic spectra with the flagging
3) Normalize the bandpass using a subband-wise spline fit
4) Remove obvious outliers
5) Plot (and save) the dynamic spectra with the flattened bandpass
6) Save those spectra as filterbank files

PB 2025
'''

# Import all necessary modules
import os
import your
import argparse
import numpy as np
import multiprocessing as mp

from file_handling import get_data, save_to_fil, assign_calibration_by_order
from plot_bursts import plot_dynspec
from bandpass_correction import correct_bandpass_spline
from spectrum_functions import flag_rfi

def parse_args():
	'''
	Create and return the Argument Parse namespace with all arguments
	'''
	
	parser = argparse.ArgumentParser(
		description="Input arguments usage: file_path"
	)
	
	# Required argument: input file directory
	parser.add_argument(
		"--file_path", "-i",
		type=str,
		required=True, 
		help="Path to the input directory where the data is stored"
	)
	
	parser.add_argument(
		"--obs_name", "-n", 
		type=str,
		required=True,
		help="Name of the observation set"
	)
	
	parser.add_argument(
		"--plot_path_1", "-p",
		type=str,
		required=True,
		help="Path to the plotting directory for the AO-flagged dynamical spectra"
	)
	
	parser.add_argument(
		"--diag_path", "-d",
		type=str,
		required=True,
		help="Path where the diagnostic spline fit plots are saved"
	)
	
	parser.add_argument(
		"--plot_path_2", "-o",
		type=str,
		required=True,
		help="Path to the directory for the dynamic spectra plots after spline bandpass correction"
	)
	
	parser.add_argument(
		"--save_path", "-s",
		type=str,
		required=True, 
		help="Path to the directory where the bandpass corrected filterbank files are saved"
	)
		
	return parser.parse_args()
		
def process(idx, fname, obs_name, files):

	args = parse_args()
	
	# Accessing arguments
	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	plot_path_1 = args.plot_path_1		# Directory for the AO-flagged plots
	diag_path = args.diag_path		# Directory for the spline diagnostic plots
	plot_path_2 = args.plot_path_2		# Directory for the spline flattened dynspec plots	
	save_path = args.save_path		# Directory for the save filterbank files

	fnum = os.path.basename(fname)[0:2]
		
	# Create a Your Object
	your_object = your.Your(fname)
		
	# read out the header of the your file
	hdr = your_object.your_header
		
	source_name = hdr.source_name		# Name of the source
	tsamp = hdr.tsamp			# Sample time
	nsamp = hdr.nspectra			# Number of time samples
		
	if fname.endswith("_I.fil"):
		stokes = 'I'
	elif fname.endswith("_V.fil"):
		stokes = 'V'
		
	# Print out the source that's being analysed
	print('{}/{} - Source: {}'.format(idx+1, len(files), source_name))
		
	data = get_data(your_object, nstart=0, nsamp=nsamp, npoln=1, keys='infer')
		
	
	# Plotting the Flagged data plots 
	plot_name = "{}_{}_{}_{}".format(fnum, obs_name, source_name, stokes)
	plot_dynspec(data, hdr, 0, nsamp, tsamp, obs_name, stokes, 'min', idx+1, hdr.tstart_utc, plot_path_1, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
		
	full_diag_path = '{}/{}'.format(diag_path, plot_name)
	data_SPL = correct_bandpass_spline(data.copy(), full_diag_path, 8, 1, False)
	
	data_flag = flag_rfi(data_SPL.copy(), 3)
	
	# Plotting the bandpass corrected data
	plot_dynspec(data_flag, hdr, 0, nsamp, tsamp, obs_name, stokes, 'min', idx+1, hdr.tstart_utc, plot_path_2, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
			
	# Save as filterbank files
	save_to_fil(data_flag, hdr, obs_name, 'I', np.shape(data_flag)[0], 0, tsamp, save_path, plot_name)


def main():
	'''
	Main entry point of the script.
	
	Uses parset command-line arguments.
	'''
	
	args = parse_args()
	
	# Accessing arguments
	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	plot_path_1 = args.plot_path_1		# Directory for the AO-flagged plots
	diag_path = args.diag_path		# Directory for the spline diagnostic plots
	plot_path_2 = args.plot_path_2		# Directory for the spline flattened dynspec plots	
	save_path = args.save_path		# Directory for the save filterbank files
	
#	print(file_path)
	# Create a list of all filterbank files in the given directoy 
	files = [file for file in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, file))]
	
	# Sort the files alphabetically (in order of observation)
	files = sorted(files)
	
	# Remove the .ar and .ps files from the filename array
	files = [file for file in files if file.endswith('.fil')]
		
	calibration_map, source_files, pulsar_files, source_counts = assign_calibration_by_order(files, file_path)
	
	# Binning up the pulsar files
	for idx, fname in enumerate(source_files):
		
		print('--------')	
		print(fname)
		
		p = mp.Process(
			target=process,
			args=(idx, fname, obs_name, files)
		)
		
		p.start()
		p.join()		


if __name__ == "__main__":
	main()
