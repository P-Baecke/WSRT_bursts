'''
This script is used to 
1) read in raw WSRT single dish filterbank files
2) calculate the stokes parameters I, V
3) plot (and save) the raw dynamic spectra for Stokes I and V
4) Remove subbands 4 and 5
5) Remove subband edges
6) Remove startup irregularities
7) Save those Stokes I and V files as filterbank files

PB 2025
'''

# Import all necessary modules
import os
import sys
sys.path.append('/root/Research_Project')
import your
import argparse
import numpy as np
import multiprocessing as mp

from file_handling import assign_calibration_by_order, get_data, save_to_fil
from spectrum_functions import mask_subband_borders, subband_flag, flag_startup
from plot_bursts import plot_dynspec

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
		help="Path to the working directory where the data is stored"
	)
	
	parser.add_argument(
		"--obs_name", "-n", 
		type=str,
		required=True,
		help="Name of the observation set"
	)
	
	parser.add_argument(
		"--plot_path", "-p",
		type=str,
		required=True,
		help="Path to the plotting directory for the raw Stokes I and V plots"
	)
	
	parser.add_argument(
		"--save_path", "-s",
		type=str,
		required=True,
		help="Path where the new filterbanks are being saved"
	)
	
	parser.add_argument(
		"--diag_path", "-d",
		type=str,
		required=True,
		help="Path to the directory for the startup flagging diagnostic plot"
	)
		
	return parser.parse_args()
		
		
def process(idx, fname, obs_name, source_files):
	args = parse_args()
	
	# Accessing arguments
	obs_name = args.obs_name		# Observation name
	plot_path = args.plot_path		# Directory for the dynamic spectrum plots
	save_path = args.save_path		# Directory for the save filterbank files
	diag_path = args.diag_path		# Directory for the diagnostic plots

	# Create a Your Object
	your_object = your.Your(fname)
		
	# read out the header of the your file
	hdr = your_object.your_header
		
	source_name = hdr.source_name		# Name of the source
	tsamp = hdr.tsamp			# Sample time
	nsamp = hdr.nspectra			# Number of time samples
		
	# Print out the source that's being analysed
	print('{}/{} - Source: {}'.format(idx+1, len(source_files), source_name))

	# Read in data dictionary
	data = get_data(your_object, nstart=0, nsamp=nsamp, npoln=4, keys='infer')
		
	# Initiate the 2D array for the desired polarization
	LL_data = data['LL']
	RR_data = data['RR']
		
	# Forming the Stokes Parameters
	print('Calculating Stokes I and V data')
	I_data = LL_data + RR_data
	V_data = LL_data - RR_data
		
	if idx+1 < 10:
		plot_name = "0{}_{}_{}".format(idx+1, obs_name, source_name)
	else:
		plot_name = "{}_{}_{}".format(idx+1, obs_name, source_name)
			
	# Saving the raw data dynamic spectra plots for Stokes I
	plot_dynspec(I_data, hdr, 0, nsamp, tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path, f'{plot_name}_I', freq_start_channel=0, freq_end_channel=None, savefig=True)
		
	# Saving the raw data dynamic spectra plots for Stokes V
	plot_dynspec(V_data, hdr, 0, nsamp, tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path, f'{plot_name}_V', freq_start_channel=0, freq_end_channel=None, savefig=True)

	# Flag the edges of subbands
	I_data = mask_subband_borders(I_data.copy(), 8, border_width=5)
	V_data = mask_subband_borders(V_data.copy(), 8, border_width=5)
		
	# Flag large subband chunks
	I_data = subband_flag(I_data.copy(), hdr)
	V_data = subband_flag(V_data.copy(), hdr)
		
	# Flag startup irregularities
	I_data = flag_startup(I_data.copy(), hdr, 'I', tsamp, diag_path, f'{plot_name}_I')
	V_data = flag_startup(V_data.copy(), hdr, 'V', tsamp, diag_path, f'{plot_name}_V')
	
	# Saving the Stokes I and V files as filterbank files
	save_to_fil(I_data, hdr, obs_name, 'I', np.shape(I_data)[0], 0, tsamp, save_path, f'{plot_name}_I')
	save_to_fil(V_data, hdr, obs_name, 'V', np.shape(V_data)[0], 0, tsamp, save_path, f'{plot_name}_V')	


def main():
	'''
	Main entry point of the script.
	
	Uses parset command-line arguments.
	'''
	
	args = parse_args()
	
	# Accessing arguments
	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	
	# Create a list of all filterbank files in the given directoy 
	files = [file for file in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, file))]
	
	# Sort the files alphabetically (in order of observation)
	files = sorted(files)
	
	# Remove the .ar and .ps files from the filename array
	files = [file for file in files if file.endswith('.fil')]
	
	# Assign sources to calibration pulsars
	calibration_map, source_files, pulsar_files, source_counts = assign_calibration_by_order(files, file_path)
	
	# Binning up the pulsar files
	for idx, fname in enumerate(source_files):

		print('-----')	
		print(fname)

		p = mp.Process(
			target=process,
			args=(idx, fname, obs_name, source_files)
		)
		
		p.start()
		p.join()

if __name__ == "__main__":
	main()
