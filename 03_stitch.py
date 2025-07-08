'''
This script is used to 
1) Read in triple flagged bandpass corrected filterbank files
2) Only select subbands 0, 1, and 2
3) Plot (and save) the dynamic spectra
4) Remove sharp RFI spikes on the time axis
5) Stitch all files of one Stokes Parameter of one source together
6) Adjust for time variations using a spline fit
7) Plot (and save) the resulting full-observation plot (full res working dir)
8) Bin up in time
9) Plot (and save) the resulting full-observation plot (lower res home dir)
8) Save the resulting filterbank files (full res)

PB 2025
'''

# Import all necessary modules
import os
import gc
import your
import argparse
import numpy as np
import multiprocessing as mp

from file_handling import get_data, save_to_fil, assign_calibration_by_order
from plot_bursts import plot_dynspec
from bandpass_correction import time_spline
from spectrum_functions import stitch_observations, select_subbands, flag_time_spikes, bin_time

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
		"--plot_path_1", "-p1",
		type=str,
		required=True,
		help="Path to where the triple flagged single observations plots are saved"
	)
	
	parser.add_argument(
		"--diag_path", "-d",
		type=str,
		required=True,
		help="Directory of the diagnostic time rfi removal plots"
	)
	
	parser.add_argument(
		"--diag_path_2", "-d2",
		type=str,
		required=True,
		help="Directory of the diagnostic full spline plot"
	)
	
	parser.add_argument(
		"--plot_path_2", "-p2",
		type=str,
		required=True,
		help="Directory of the full time dynamic spectra"
	)
	
	parser.add_argument(
		"--plot_path_3", "-p3",
		type=str,
		required=True,
		help="Directory of the binned-up total time dynamic spectra (home directory)"
	)
		
	parser.add_argument(
		"--save_path", "-s",
		type=str,
		required=True,
		help="Directory where to save the entire observation set as filterbank file"
	)
	
	parser.add_argument(
		"--tvar_path", "-t",
		type=str,
		required=True,
		help="Directory where to save the single observation sets, time variance corrected as dynamic spectra plots"
	)
	
	parser.add_argument(
		"--tvar_path_2", "-t2",
		type=str,
		required=True,
		help="Directory where to save the single observation sets, time variance corrected as filterbank files"
	)
		
	parser.add_argument(
		"--tvar_path_3", "-t3",
		type=str,
		required=True,
		help="Directory where the time variation diagnostic plots are saved."
	)
	
	return parser.parse_args()
		
		
		
def process_I_files(idx, fname, obs_name, I_files):

	args = parse_args()

	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	plot_path_1 = args.plot_path_1		# Directory to where the single dynamic spectra shall be saved
	diag_path = args.diag_path		# Directory of the diagnostic time rfi plots
	diag_path_2 = args.diag_path_2		# Directory of the total time spline diagnostic
	plot_path_2 = args.plot_path_2		# Directory of the total time dynamic spectra
	plot_path_3 = args.plot_path_3		# Directory of the binned-up total time dynamic spectra (home directory)
	save_path = args.save_path		# Directory where to save the entire observation set as filterbank file
	tvar_path = args.tvar_path		# Directory of the single observation time variation corrected dynamic spectra
	tvar_path_2 = args.tvar_path_2		# Directory of the single observation time variation corrected dynamic spectra filterbank files
	tvar_path_3 = args.tvar_path_3	# Directory where the time variation diagnostic plots are saved
	
	fnum = os.path.basename(fname)[0:2]

	# Create a Your Object
	your_object = your.Your(fname)
		
	# read out the header of the your file
	hdr = your_object.your_header
		
	source_name = hdr.source_name		# Name of the source
	tsamp = hdr.tsamp			# Sample time
	nsamp = hdr.nspectra			# Number of time samples

							
	# Print out the source that's being analysed
	print('\n{}/{} - Source: {}'.format(idx+1, len(I_files), source_name))
		
	data = get_data(your_object, nstart=0, nsamp=nsamp, npoln=1, keys='infer')

	# Only look at the first three subbands
	# slice_data, start_channel, end_channel = select_subbands(data.copy(), 8, [0, 1, 2])
		
	# nchans = end_channel - start_channel
	start_channel = 0
	end_channel = np.shape(data)[0]-1
	nchans = np.shape(data)[0]

	# Plotting those
	plot_name = "{}_{}_{}_I".format(fnum, obs_name, source_name)		
	plot_dynspec(data, hdr, 0, nsamp, tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path_1, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
		
		
	full_diag_path = '{}/{}'.format(diag_path, plot_name)
	# Remove sharp RFI spikes on the time axis
	data_flag = flag_time_spikes(data, hdr, 'I', tsamp, full_diag_path)
		
	full_tvar_path = '{}/{}'.format(tvar_path_3, plot_name)
	single_data_time = time_spline(data_flag, tsamp, full_tvar_path, n_subbands=8, smooth_param=250, return_spline=False)
		
	plot_dynspec(single_data_time, hdr, 0, np.shape(single_data_time)[1], tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, tvar_path, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
		
	save_to_fil(single_data_time, hdr, obs_name, 'I', np.shape(single_data_time)[0], 0, tsamp, tvar_path_2, plot_name) 
		
def process_V_files(idx, fname, obs_name, V_files):

	args = parse_args()

	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	plot_path_1 = args.plot_path_1		# Directory to where the single dynamic spectra shall be saved
	diag_path = args.diag_path		# Directory of the diagnostic time rfi plots
	diag_path_2 = args.diag_path_2		# Directory of the total time spline diagnostic
	plot_path_2 = args.plot_path_2		# Directory of the total time dynamic spectra
	plot_path_3 = args.plot_path_3		# Directory of the binned-up total time dynamic spectra (home directory)
	save_path = args.save_path		# Directory where to save the entire observation set as filterbank file
	tvar_path = args.tvar_path		# Directory of the single observation time variation corrected dynamic spectra
	tvar_path_2 = args.tvar_path_2		# Directory of the single observation time variation corrected dynamic spectra filterbank files
	tvar_path_3 = args.tvar_path_3	# Directory where the time variation diagnostic plots are saved

	fnum = os.path.basename(fname)[0:2]
	# Create a Your Object
	your_object = your.Your(fname)
		
	# read out the header of the your file
	hdr = your_object.your_header
		
	source_name = hdr.source_name		# Name of the source
	tsamp = hdr.tsamp			# Sample time
	nsamp = hdr.nspectra			# Number of time samples

							
	# Print out the source that's being analysed
	print('\n{}/{} - Source: {}'.format(idx+1, len(V_files), source_name))
		
	data = get_data(your_object, nstart=0, nsamp=nsamp, npoln=1, keys='infer')

	# Only look at the first three subbands
	# slice_data, start_channel, end_channel = select_subbands(data.copy(), 8, [0, 1, 2])
		
	start_channel = 0
	end_channel = np.shape(data)[0]-1
	
	nchans = np.shape(data)[0]


	# Plotting those
	plot_name = "{}_{}_{}_V".format(fnum, obs_name, source_name)		
	plot_dynspec(data, hdr, 0, nsamp, tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path_1, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
		

	full_diag_path = '{}/{}'.format(diag_path, plot_name)
	# Remove sharp RFI spikes on the time axis
	data_flag = flag_time_spikes(data.copy(), hdr, 'V', tsamp, full_diag_path)
	
	full_tvar_path = '{}/{}'.format(tvar_path_3, plot_name)
	single_data_time = time_spline(data_flag.copy(), tsamp, full_tvar_path, n_subbands=8, smooth_param=250, return_spline=False)

	plot_dynspec(single_data_time, hdr, 0, np.shape(single_data_time)[1], tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, tvar_path, plot_name, freq_start_channel=0, freq_end_channel=None, savefig=True)
		
	save_to_fil(single_data_time, hdr, obs_name, 'V', np.shape(single_data_time)[0], 0, tsamp, tvar_path_2, plot_name)


def main():
	'''
	Main entry point of the script.
	
	Uses parset command-line arguments.
	'''
	
	args = parse_args()
	
	# Accessing arguments
	file_path = args.file_path		# Directory of the original files
	obs_name = args.obs_name		# Observation name
	plot_path_1 = args.plot_path_1		# Directory to where the single dynamic spectra shall be saved
	diag_path = args.diag_path		# Directory of the diagnostic time rfi plots
	diag_path_2 = args.diag_path_2		# Directory of the total time spline diagnostic
	plot_path_2 = args.plot_path_2		# Directory of the total time dynamic spectra
	plot_path_3 = args.plot_path_3		# Directory of the binned-up total time dynamic spectra (home directory)
	save_path = args.save_path		# Directory where to save the entire observation set as filterbank file
	tvar_path = args.tvar_path		# Directory of the single observation time variation corrected dynamic spectra
	tvar_path_2 = args.tvar_path_2		# Directory of the single observation time variation corrected dynamic spectra filterbank files
	tvar_path_3 = args.tvar_path_3	# Directory where the time variation diagnostic plots are saved
	
	
	# Create a list of all filterbank files in the given directoy 
	files = [file for file in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, file))]
	
	# Sort the files alphabetically (in order of observation)
	files = sorted(files)
	
	calibration_map, source_files, pulsar_files, source_counts = assign_calibration_by_order(files, file_path)
	
	# Remove the .ar and .ps files from the filename array
	# Immediatly separate into Stokes I and V files
	I_files = [file for file in source_files if file.endswith('_I.fil')]
	V_files = [file for file in source_files if file.endswith('_V.fil')]
	
	# Initiate variable for the following loops
	current_source = None
	total_data = None
	
	# Performing all steps of stitching and RFI removal for Stokes I files
	
	for idx, fname in enumerate(I_files):
	
		print('-----')
		print(fname)
		
		p = mp.Process(
			target=process_I_files,
			args=(idx, fname, obs_name, I_files)
		)
		
		p.start()
		p.join()
	
	
	
		
	'''
		# Stitching logic
		if current_source is None:
			# First file overall
			current_source = source_name
			total_data = None
			
		elif source_name != current_source:
			# Source changed: finalize the stitched block for the old source
			# Time Spline
			
			full_name = '{}_{}_I'.format(obs_name, current_source)
			full_diag_path_2 = '{}/{}'.format(diag_path_2, full_name)
			data_time = time_spline(total_data.copy(), tsamp, full_diag_path_2, n_subbands=8, smooth_param=250, return_spline=False)
			
			# Plot full resolution plot into working directory
			plot_dynspec(data_time, hdr, 0, np.shape(data_time)[1], tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path_2, full_name, start_channel, end_channel, savefig=True)
			
			# Bin up in time to allow plots to be quickly displayable in the home directory
			new_tsamp = tsamp * 64		# Binning up to 1.048576s per timestamp
			data_bin = bin_time(data_time, hdr, tsamp, new_tsamp)
			
			# Plot lower resolution plot into home directory
			plot_dynspec(data_bin, hdr, 0, np.shape(data_bin)[1], new_tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path_3, full_name, start_channel, end_channel, savefig=True)
	

			# Save as filterbank files
			save_to_fil(data_time, hdr, obs_name, 'I', np.shape(data_time)[0], 0, tsamp, save_path, full_name)
			 


			# Reset for the new source
			current_source = source_name
			total_data = None
		
		# Stitch the current file's data into total_data
		if total_data is None:
			total_data, total_nsamp = stitch_observations(total_data, data_flag.copy(), nchans, tsamp)
		elif total_data is not None:
			total_data, total_nsamp = stitch_observations(total_data.copy(), data_flag.copy(), nchans, tsamp)
	'''
		
	
	'''
	# After the loop ends, finalize the last source
	if current_source is not None and total_data is not None:
		# plot working dir
		# plot home dir
		# Save as filterbank file
		# Normalize time variations using a time spline fit
		full_name = '{}_{}_I'.format(obs_name, current_source)
		full_diag_path_2 = '{}/{}'.format(diag_path_2, full_name)
		data_time = time_spline(total_data.copy(), tsamp, full_diag_path_2, n_subbands=8, smooth_param=250, return_spline=False)
			
		# Plot full resolution plot into working directory
		plot_dynspec(data_time, hdr, 0, np.shape(data_time)[1], tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path_2, full_name, start_channel, end_channel, savefig=True)
			
		# Bin up in time to allow plots to be quickly displayable in the home directory
		new_tsamp = tsamp * 64		# Binning up to 1.048576s per timestamp
		data_bin = bin_time(data_time, hdr, tsamp, new_tsamp)
			
		# Plot lower resolution plot into home directory
		plot_dynspec(data_bin, hdr, 0, np.shape(data_bin)[1], new_tsamp, obs_name, 'I', 'min', idx+1, hdr.tstart_utc, plot_path_3, full_name, start_channel, end_channel, savefig=True)
	

		# Save as filterbank files
		save_to_fil(data_time, hdr, obs_name, 'I', np.shape(data_time)[0], 0, tsamp, save_path, full_name)

	'''

	# Initiate variable for the following loops
	current_source = None
	total_data = None
	
	# Performing all steps of stitching and RFI removal for Stokes V files
	for idx, fname in enumerate(V_files):
	
		print('-----')
		print(fname)

		p = mp.Process(
			target=process_V_files,
			args=(idx, fname, obs_name, I_files)
		)
		
		p.start()
		p.join()

		'''	
		# Stitching logic
		if current_source is None:
			# First file overall
			current_source = source_name
			total_data = None
			
		elif source_name != current_source:
			# Source changed: finalize the stitched block for the old source
			# Time Spline
			
			full_name = '{}_{}_V'.format(obs_name, current_source)
			full_diag_path_2 = '{}/{}'.format(diag_path_2, full_name)
			data_time = time_spline(total_data.copy(), tsamp, full_diag_path_2, n_subbands=8, smooth_param=250, return_spline=False)
			
			# Plot full resolution plot into working directory
			plot_dynspec(data_time, hdr, 0, np.shape(data_time)[1], tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path_2, full_name, start_channel, end_channel, savefig=True)
			
			# Bin up in time to allow plots to be quickly displayable in the home directory
			new_tsamp = tsamp * 64		# Binning up to 1.048576s per timestamp
			data_bin = bin_time(data_time, hdr, tsamp, new_tsamp)
			
			# Plot lower resolution plot into home directory
			plot_dynspec(data_bin, hdr, 0, np.shape(data_bin)[1], new_tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path_3, full_name, start_channel, end_channel, savefig=True)
	

			# Save as filterbank files
			save_to_fil(data_time, hdr, obs_name, 'V', np.shape(data_time)[0], 0, tsamp, save_path, full_name)


			# Reset for the new source
			current_source = source_name
			total_data = None
		
		# Stitch the current file's data into total_data
		if total_data is None:
			total_data, total_nsamp = stitch_observations(total_data, data_flag.copy(), nchans, tsamp)
		elif total_data is not None:
			total_data, total_nsamp = stitch_observations(total_data.copy(), data_flag.copy(), nchans, tsamp)
		'''
		
	'''		
	# After the loop ends, finalize the last source
	if current_source is not None and total_data is not None:
		# plot working dir
		# plot home dir
		# Save as filterbank file
		# Normalize time variations using a time spline fit
		full_name = '{}_{}_V'.format(obs_name, current_source)
		full_diag_path_2 = '{}/{}'.format(diag_path_2, full_name)
		data_time = time_spline(total_data.copy(), tsamp, full_diag_path_2, n_subbands=3, smooth_param=250, return_spline=False)
			
		# Plot full resolution plot into working directory
		plot_dynspec(data_time, hdr, 0, np.shape(data_time)[1], tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path_2, full_name, start_channel, end_channel, savefig=True)
			
		# Bin up in time to allow plots to be quickly displayable in the home directory
		new_tsamp = tsamp * 64		# Binning up to 1.048576s per timestamp
		data_bin = bin_time(data_time, hdr, tsamp, new_tsamp)
			
		# Plot lower resolution plot into home directory
		plot_dynspec(data_bin, hdr, 0, np.shape(data_bin)[1], new_tsamp, obs_name, 'V', 'min', idx+1, hdr.tstart_utc, plot_path_3, full_name, start_channel, end_channel, savefig=True)
	

		# Save as filterbank files
		save_to_fil(data_time, hdr, obs_name, 'V', np.shape(data_time)[0], 0, tsamp, save_path, full_name)
	'''
	
if __name__ == "__main__":
	main()
