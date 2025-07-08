'''
This function contains all array manipulation functions for the radio burst script.

PB 2025
'''

#Import all necessary modules
import numpy as np
import your
import sys
from astropy.io import fits
# from your.candidate import Candidate
# from your.utils.plotter import plot_h5
from scipy.signal import detrend, savgol_filter
import os
import struct
import mmap
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from astropy.coordinates import Angle, SkyCoord, EarthLocation
from astropy import units as u
from astropy.time import Time
from scipy.interpolate import interp1d, UnivariateSpline
import scipy.interpolate as interp
import tempfile
import subprocess
import matplotlib.pyplot as plt


def dedisperse(data, hdr, tsamp, pol_name):
	'''
	Dedisperse a 2D array of time-frequency data
	Inputs:
		data (2D array)     :	2D array containing the time-freq data
		hdr (your.hdr)      :	Your header of the corresponding data
		tsamp (s)  	    :	Seconds that pass per sample 
		pol_name (str)      :	Name of the choosen polarization (LL, RR, LR, RL)
	Returns:
		data_dedisp (arr):	2D time vs frequency array of the dedispersed dynamic spectrum

	'''

	source_name = hdr.source_name			# name of the source

	print('Dedispersing {} data of Pulsar {}'.format(pol_name, source_name))


	# https://academic.oup.com/mnras/article/353/4/1311/977600?login=true
	if source_name == 'B0329+54':
		# https://www.aanda.org/articles/aa/pdf/2012/07/aa18970-12.pdf
		DM = 26.833		# Dispersion measure of that pulsar in pc cm-3
		nu = 1.39954153872		# Pulse frequency of that pulsar in 1/s
		nu_dot = -4.011970*1e-15	# Pulsar spindown in 1/s²
		epoch = 46473.0 		# Epoch of the period in MJD
	elif source_name == 'B1933+16':
		# https://iopscience.iop.org/article/10.3847/1538-4357/ac9eae/pdf
		DM = 158.521		# Dispersion measure of that pulsar in pc cm-3
		nu = 2.78754649621		# Pulse frequency of that pulsar in 1/s
		nu_dot = -46.642103*1e-15	# Pulsar spindown in 1/s²
		epoch = 46434.0 		# Epoch of the period in MJD
	else:
		raise ValueError("Source {} is not supported".format(source_name))

	t_MJD = hdr.tstart				# MJD time stamp of the first sample
	t_dif = (t_MJD - epoch) * 86400	# Difference between parameter epoch and observation in s

	P = 1.0 / (nu + nu_dot*t_dif)		# Pulse period in seconds

	'''
	if source_name == 'B0329+54':
		DM = 26.833		# Dispersion measure of that pulsar in pc cm-3
		nu = 1.39954153872	# Pulse frequency of that pulsar in 1/s
		# https://academic.oup.com/mnras/article/353/4/1311/977600?login=true
		# https://www.aanda.org/articles/aa/pdf/2012/07/aa18970-12.pdf
		
	elif source_name == 'B1933+16':
		DM = 158.521		# Dispersion measure of that pulsar in pc cm-3
		nu = 2.78754649621	# Pulse frequency of that pulsar in 1/s
		# https://iopscience.iop.org/article/10.3847/1538-4357/ac9eae/pdf
		# https://academic.oup.com/mnras/article/353/4/1311/977600?login=true

	

	P = 1.0 / nu		# Pulse period in seconds
	'''
	
	print('DM = {} pc cm⁻³'.format(DM))
	
	# Create an array containing the frequency channels
	freq = hdr.fch1 + np.arange(hdr.nchans) * hdr.foff

	# Calculate the frequency channels' delays
	delay = np.zeros(data.shape[0])		# Array to store the delay for each frequency channel
	for i in range(data.shape[0]):
		delay[i] = 4.15e3 * DM * (1/freq[0]**2 - 1/freq[i]**2)
		# Following Lorimer and Kramer eq (5.1)

	# Convert the delay to samples
	delay_samples = np.round(delay/tsamp).astype(int)
	# print(delay_samples)

	# Create a new array for the dedispersed data
	dedisp_data = np.zeros_like(data)

	# Apply the delay by shifting each frequency channel
	for i in range(data.shape[0]):
		dedisp_data[i, :] = np.roll(data[i, :], delay_samples[i])

	return dedisp_data


def fold_pulsar_data_phase_frac(data, hdr, t_bary, pol_name, nbin=250):
	'''
	Fold pulsar data into phase bins using high-resolution fractional-bin weightin, then optionally re-bin down to a coarser grid.
	
	This avoids phase-quantization errors by computing each sample's exact phase-bin index and splitting it's weight between the two nearest bins.
	
	Parameters:
		data (2D array)	:	2D array containing the dynamic spectrum
		hdr (your object)	:	Header of the your file containing observation metadata
		t_bary (1D array)	:	Barycentric timestamps for each sample (TDB scale)
		pol_name (str)		:	String indicating the polarization
		nbin (int)		:	Number of pahse bins to fold into (default=250)
		
	Returns:
		folded_data (2D arr)	:	2D array of dimensions nchans x nbin containing the folded pulsar data
		nbin (int)		:	Number of samples (phase bins) per pulse period
		psr_tsamp (int)	:	Number of seconds that pass per sample
		N_pulses (int)		:	Total nuumber of pulses used in the folding
	'''
	
	source_name = hdr.source_name
	nchan = hdr.nchans
	        
	# Choose the pulsar period based on the source name
	# Value taken from:
	# https://academic.oup.com/mnras/article/353/4/1311/977600?login=true
	if source_name == 'B0329+54':
		nu = 1.39954153872		# Pulse frequency of that pulsar in 1/s
		nu_dot = -4.011970*1e-15	# Pulsar spindown in 1/s²
		epoch = 46473.0 		# Epoch of the period in MJD
	elif source_name == 'B1933+16':
		nu = 2.78754649621		# Pulse frequency of that pulsar in 1/s
		nu_dot = -46.642103*1e-15	# Pulsar spindown in 1/s²
		epoch = 46434.0 		# Epoch of the period in MJD
	else:
		raise ValueError("Source {} is not supported for pulse folding".format(source_name))
		
	t_MJD = t_bary[0].mjd			# MJD time stamp of the first sample
	t_dif = (t_MJD - epoch) * 86400	# Difference between parameter epoch and observation in s
	P = 1.0 / (nu + nu_dot*t_dif)		# Pulse period in seconds
	# P = 1.0 / nu

	print('Folding {} data for {} with P={}s'.format(pol_name, source_name, round(P, 4)))

	# Compute total observation span and pulse count
	tsec = (t_bary.unix - t_bary.unix[0])		# Seconds since the first sample
	total_span = tsec[-1]				# Length of observation set in seconds
	N_pulses = int(np.floor(total_span / P))
	
	# Phase for each sample in [0, 1)
	# phase = (1.0 - ((tsec % P) / P)) % 1.0
	phase = (tsec % P) / P
	
	# Initialize folded profile and counters
	folded_data = np.zeros((nchan, nbin), dtype=hdr.dtype)
	counts = np.zeros((nchan, nbin), dtype=hdr.dtype)
	
	# Compute the floating bin index and its floor and fractional part
	float_idx = phase * nbin		# [0, nbin)
	k = np.floor(float_idx).astype(int)	# lower bin index
	f = float_idx - k			# fractional weight
	k1 = (k+1) % nbin			# upper bin index (wrap)

	# Distribute each sample into the two nearest bins
	for itime in range(data.shape[1]):
	
		# build boolean mask of non-NaN entries
		mask = ~np.isnan(data[:, itime])
		d = data[mask, itime]
		
        # weight into bin k with (1-f) and into k1 with f
		folded_data[mask, k[itime]] += d * (1.0-f[itime])
		folded_data[mask, k1[itime]] += d * f[itime]
		counts[mask, k[itime]] += (1.0 -f[itime])
		counts[mask, k1[itime]] += f[itime]	
	
	
	# Avoid division by zero: any bin with zero counts set to one
	counts[counts == 0] = 1
	
	# Normalize folded profile by number of contributions per bin
	folded_data = folded_data / counts

	folded_data[folded_data == 0] = np.nan

	# Sample time for bin size
	psr_tsamp = P / nbin
	
	print(f"Folding complete. Estimated full pulses: {N_pulses}")
	
	return folded_data, nbin, psr_tsamp, N_pulses

def bin_time(data, hdr, old_tsamp, new_tsamp):
	'''
	This function bins up the data on the time axis over the entire observation
	Inputs:
		data (2D array)	:	2D array containing the unbinned data
		hdr (your.hdr)		:	Your header of the corresponding data  
		old_tsamp (s)		:	Seconds that pass per sample for the old bin size
		new_tsamp (s)		:	Seconds that pass per sample for the new bin size
	Returns:
		binned_data (2D array)	:	2D array containing the binned data
	'''


	print('Binning the data into {} second blocks'.format(new_tsamp))

	# Determine how many samples are included in a bin
	bin_nsamp = round(new_tsamp/old_tsamp)

	# Calculate the totla number of bins
	num_bins = hdr.nspectra//bin_nsamp

	# Initiating the array for the new data
	binned_data = np.zeros((hdr.nchans, num_bins), dtype=hdr.dtype)

	# Iterate over each bin and calculate the average
	for i in range(num_bins):
		start_idx = i * bin_nsamp
		end_idx = start_idx + bin_nsamp
		binned_data[:, i] = np.nanmean(data[:, start_idx:end_idx], axis=1)

	# return
	return binned_data

def stitch_observations(total_data, input_data, nchans, tsamp):
	'''
	This function is meant to take two blocks of data, and concatenated them with a defined gap.
	Inputs:
		total_data (2D array)	:	The accumulated data array
		input_data (2D array)	:	The new data block to append
		nchans (int)		:	Number of frequency channels
		tsamp (int)		:	Sampling time in seconds
	Returns:
		conc_data (2D array)	:	2D array containing the concatenated data

	'''

	# Create the gap array describing gap inbetween observations
	gap_sec = 2		# The gap between two observations on the same source is two seconds with WSRT RT2
	gap = np.full((nchans, int(gap_sec/tsamp)), np.nan, dtype=float)

	# gap = np.isnan((nchans, int(gap_sec/new_tsamp)))

	if total_data is None:
		# Initialize total_dara if it's the first block
		total_data = input_data.copy()
	else:
		# Append the gap and input data to the total_data
		total_data = np.concatenate((total_data, gap, input_data), axis=1, dtype='float32')

	# Number of samples on the new total data array
	new_nsamp = np.shape(total_data)[1]

	return total_data, new_nsamp


def mask_subband_borders(data, num_subbands=8, border_width=5):
	''' 
	Mask channels around subband borders in a 2D dynamic spectrum array.
	
	This function identifies the bordesr between subbands and set a specified number of channels on each side of each border to NaN to clean subband edges.
	
	Inputs:
		data (2D array)	:	Input dynamic spectrum array
		num_subbands (int)	:	Total number of subbands in the data, default is 8
		border_width (int)	:	The number of channels to mask on each side of each border. Default is 5
		
	Returns:
		data (2D)	:	A copy of the input data with the channels around each subband border set to NaN
	'''
	
	# Calculate the total number of frequency channels
	nchans = np.shape(data)[0]
	
	# Calculate the subband width
	subband_width = nchans // num_subbands
	
	# Compute the indices where subbands meet
	border_positions = [i * subband_width for i in range(num_subbands+1)]
	
	# Calculate the mask and set to NaN
	for pos in border_positions:
		# Define the slice to mask, clipped to [0, nchans)
		start = max(pos - border_width, 0)
		end = min(pos + border_width, nchans)
		
		# Set all the time samples in these frequency channels to NaN
		data[start:end, :] = np.nan
		
	return data
	
def subband_flag(data, hdr, n_subbands=8, flag_band=[2, 3, 4]):
	'''
	Flags a large chunk of rfi-infested frequency channels from the dynamic spectrum.
	
	Inputs:
		data (2D array)	:	Input dynamic spectrum array (nchan, nsamp)
		hdr (your_header)	:	Header of the data file
		n_subbands (int)	:	Total number of subbands the frequency range is divided into (optional)
		flag_band (list[int])	:	Indices of subbands to flag (0-based)
	
	Returns:
		data (2D array)	:	The dynamic spectrum with the unwanted channels removed
	'''
	
	nchan = np.shape(data)[0]		# Number of frequency channels)
	
	# Calculate number of channels per subband
	subband_size = nchan // n_subbands
	
	for band in flag_band:
		start_idx = band*subband_size
		end_idx = (band+1) * subband_size
		
		# Set all channels in the subband to NaN
		data[start_idx:end_idx, :] = np.nan
		
	# Calculate and print frequency range of the flagged subbands
	chpsb = np.shape(data)[0] / n_subbands		# Channels per subband
	start_freq = hdr.fch1 + flag_band[0] * chpsb * hdr.foff
	end_freq = hdr.fch1 + flag_band[-1] * chpsb * hdr.foff
	print('Flagged subbands {}: {} - {} MHz.'.format(flag_band, round(start_freq, 2), round(end_freq, 2)))
	
		
	return data

def select_subbands(data, n_sb, subband_indices):
	'''
	Select specified subbands from the dynamic spectrum array.
	
	Inputs: 
		data (2D array)	: 2D array containing the dynamic spectrum 
		n_sb (int)		: Total number of subbands the full data is divided into (default=8)
		subband_indices (list)	: Indices of subbands to select.
		
	Outputs:
		slided_data (2D array)	: The sliced data array containing only the selected subbands.
		start_channel (int)	: The first frequency channel index of the sliced data in the original array.
		end_channel (int)	: the last frequency channel index ü1 of the sliced data in the original array.
	'''
	
	nchans = data.shape[0]			# Total number of frequency channels
	sb_size = nchans // n_sb		# Calculate number of channels per subband (128)
	
	# Check that the subband indices are contiguous
	sorted_indices = sorted(subband_indices)
	if sorted_indices != list(range(sorted_indices[0], sorted_indices[-1] + 1)):
		raise ValueError("Subband indices ust be contiguous.")
		
	start_sb = sorted_indices[0]
	end_sb = sorted_indices[-1] +1		# +1 because slicing is exclusive at the end
	
	start_channel = start_sb * sb_size
	end_channel = end_sb * sb_size
	
	sliced_data = data[start_channel:end_channel, :]
	
	
	return sliced_data, start_channel, end_channel
		

def flag_rfi(data, threshold):
	'''
	This function calculates the time-average intensity of each frequency channel in the dynamic spectrum. It then compares that average to the overall average intensity and flags the channels that are greater than a given threshold. These channels are set to NaN

	Inputs:
		data (2D array)	: 	2D array containing the data
		threshold (float)	:	Factor, applied to the rms of the data, above which data will be flagged

	Returns:
		flag_data (2D array)	:	2D array with the data and the RFI set to NaN
	'''


	# Compute the average intensity over time for each channel
	channel_avg = np.nanmean(data, axis=1)

	# Compute overall average of the entire dynamic spectrum
	overall_avg = np.nanmean(channel_avg)

	overall_std = np.nanstd(channel_avg)

	print('Flagging RFI')
	# print('Abs Complete Average: {}'.format(overall_avg))
	# print('Maximum Value: {}'.format(np.nanmax(abs(data))))
	# print('Max BP Value: {}'.format(np.nanmax(channel_avg)))
	# print('Upper Threshold : {}'.format(overall_avg + (threshold * abs(overall_std))))
	# print('Lower Threshold : {}'.format(overall_avg - (threshold * abs(overall_std))))
	
	nchans = data.shape[0]

	flagged_channels = []

	# identify RFI channels
	for ch in range(nchans):
		if channel_avg[ch] > overall_avg + (threshold * abs(overall_std)):
			flagged_channels.append(ch)
		elif channel_avg[ch] < overall_avg - (threshold * abs(overall_std)):
			flagged_channels.append(ch)

	# Expand flagged channels to include their immediate neighbors
	expanded_flagged = set()
	for ch in flagged_channels:
		expanded_flagged.add(ch)
		if ch > 0:
			expanded_flagged.add(ch-1)
		if ch < nchans -1:
			expanded_flagged.add(ch+1)

	for ch in expanded_flagged:
		data[ch, :] = np.nan

	# return the flagged dynamic spectrum
	return data

def barycenter_dynamic_spectrum(nsamp, hdr, tsamp):
	'''
	Convert the time array of the supplied data from topocentric to barycentric time.
	
	Inputs:
		nsamp (int)		: Number of time samples in data
		hdr (your header)	: Header of the data file
		tsamp (int)		: Seconds that pass per sample
		
	Returns:
		t_bary (1D array)	: Barycentric timestanps for each sample
	'''
	
	print('Applying a barycentric correction to the time axis of {}'.format(hdr.source_name))

	# Define the telescopes's location
	location = EarthLocation(lat=52.91525*u.deg, lon=6.60247*u.deg, height=19*u.m)
	# WSRT rough location, from Google Maps
	# https://www.google.com/maps/place/52%C2%B054'54.9%22N+6%C2%B036'08.9%22E/@52.9152482,6.5999081,661m/data=!3m2!1e3!4b1!4m4!3m3!8m2!3d52.915245!4d6.602483?entry=ttu&g_ep=EgoyMDI0MDkxOC4xIKXMDSoASAFQAw%3D%3D

	# Build a topocentric time array in UTC
	t0 = Time(hdr.tstart_utc, format='isot', scale='utc', location=location)
	t_topo = t0 + np.arange(nsamp) * tsamp * u.s
	
	# Define the targe position from the file header
	target = SkyCoord(ra=hdr.ra_deg, dec=hdr.dec_deg, unit=(u.hourangle, u.deg))
	
	# Computing the light-travel time to the solar systems barycenter
	ltt_bary = t_topo.light_travel_time(target, kind='barycentric')
	
	# For barycentric times in TDB
	t_bary = t_topo.tdb + ltt_bary
	
	return t_bary

def flag_startup(data, hdr, stokes, tsamp, path, fname, threshold_sigma=3.0, buffer=5.0, max_scan=60.0):
	'''
	Detect and mask irregular startup behaviour in the dynamic spectrum.
	
	Inputs:
		data (2D array)	: Dynamic spectrum data (nchan x nsamp)
		hdr (header)		: Header object of the your file
		stokes (str)		: 'I', or 'V' indicating stokes parameter
		tsamp (float)		: Sampling time in seconds
		path (str)		: Path for the diagnostic plots to be saved
		fname (str)		: Name of the diagnostic plots
		theshold_sigma (float)	: Number of standard deviations above spline to flag
		buffer (float)		: Duration after last outlier to flag as well (in seconds)
		max_scan (float)	: Maximum duration from start to consider (in seconds)
		
	Outputs:
		data (2D array)	: Data with startup irregularities flagged out
	'''
	
	# Convert seconds to sample indices
	buffer_samples = int(buffer / tsamp)
	max_scan_samp = int(max_scan / tsamp)
	
	# compute the time-average signal
	time_avg = np.nanmean(data, axis=0)
	nsamp = len(time_avg)
	
	# Length of flagging function
	scan_limit = min(max_scan_samp, nsamp)
	x_samples = np.arange(scan_limit)
	x_seconds = x_samples * tsamp
	y = time_avg[:scan_limit]
	
	# Fit smoothing spline to the startup fraction
	spline = UnivariateSpline(x_seconds, y, s=scan_limit)
	y_fit = spline(x_seconds)
	residuals = y - y_fit
	std_resid = np.nanstd(residuals)
	
	# Find outlier beyond threshold
	outliers = np.where(np.abs(residuals) > threshold_sigma * std_resid)[0]
	
	# Require at least 10 consecutive outliers
	flagged_indices = []
	
	if len(outliers) > 0:
		# Find consecutive sequences of outliers
		diffs = np.diff(outliers)
		splits = np.where(diffs != 1)[0] + 1
		groups = np.split(outliers, splits)
		
		for group in groups:
			if len(group)>=10 :
	
				last_outlier_sample = group[-1]
				flag_limit = min(last_outlier_sample + buffer_samples, nsamp)
				flagged_indices = np.arange(0, flag_limit)
				break # only flag based on the first valid sequence
			
	else:
		flagged_indices = []
		
	# Flag data
	data[:, flagged_indices] = np.nan
	
	# Plot to check
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(np.arange(nsamp) * tsamp, time_avg, color='k', linewidth=1, label='Time-averaged Power')
	ax.plot(x_seconds, y_fit, color='blue', linestyle='--', label='Spline Fit')
	ax.plot(x_seconds, y_fit + threshold_sigma * std_resid, 'r--', linewidth=1, label='+3s')
	ax.plot(x_seconds, y_fit - threshold_sigma * std_resid, 'r--', linewidth=1, label='-3s')
	
	if len(outliers) > 0:
		ax.plot(x_seconds[outliers], y[outliers], 'ro', markersize=3, label='Outliers')
	
	ax.set_xlim(x_seconds[0] - 5, (np.arange(nsamp) * tsamp)[-1])
	ax.set_title(f'{hdr.source_name} Frequency-Averaged Spectrum (Stokes {stokes})')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Mean Power')
	ax.legend()
	
	plt.savefig(f'{path}/{fname}', bbox_inches="tight", dpi=600)
	# plt.show()
	
	plt.close()
	
	if flagged_indices != []:
		flagged_seconds = flagged_indices[-1] * tsamp
		print(f"Flagged the first {flagged_seconds:.2f} seconds due to startup irregularities.")
	else:
		print("No startup irregularities detected.")
	
	return data	
	
	
def flag_time_spikes(data, hdr, stokes, tsamp, diag_path, threshold_sigma=5.0, neighbor=3, spline_s=10000):
	'''
	Detect and mask short-duration spikes in the time series of a dynamic spectrum.
	
	This function computes the frequency-averaged time series, fits a smoothing spline to the entire series, and flags any time samples whose residuals exceed "threshold_sigma" standard deviations. In addition to each outlier, "neighbor" samples on each side are also flagged.
	
	Inputs:
		data (2D array)	: Dynamic spectrum array (nchan x nsamp)
		hdr (your header)	: Header object for the data.
		stokes (str)		: Stokes parameter label ('I', 'V')
		tsamp (float)		: Sampling time in seconds
		diag_path (str)	: Path where to save the diagnostic plot
		threshold_sigma (flt)	: Number of standard deviations above/below the spline fit beyond which is being flagged (default=5)
		neighbor (int)		: Number of time samples on each side to be flagged (default=3)
		spline_s (float)	: Smoothing factor passed to make_smoothing_spline (default=10000)
		
	Outputs:
		data_flagged (2D arr)	: Copy of the input data array with flagged samples
	'''
	data_flagged = data.copy()
	
	# Compute the frequency-average time series
	time_avg = np.nanmean(data_flagged, axis=0)
	nsamp = time_avg.size
	
	# Define the scan limit at the full series (can be adjusted if wanted)
	x_samples = np.arange(nsamp)
	x_seconds = x_samples * tsamp
	
	# Mask out NaNs to get only finite points for fitting
	finite_mask = np.isfinite(time_avg)
	x_fit = x_seconds[finite_mask]		# Time values where data is finite
	y_fit_data = time_avg[finite_mask]	# Corresponding finite averaged power
	
	# Choos smoothing parameter for the UnivariateSpline
	# If none, set it to len(y) for a relatively rough fit
	std2 = np.std(y_fit_data) ** 2
	if spline_s is None:
		spline_s = len(y_fit_data)/100.0
			
	
	# Fit a smoothing spline to the entire time series
	spline = interp.make_smoothing_spline(x_fit, y_fit_data, lam=spline_s)
	y_fit = spline(x_seconds)
		
	# COmpute residuals
	residuals = time_avg - y_fit
	std_resid = np.nanstd(residuals)
	
	# Identify all time indices where residual magnitude exceeds threshold_sigma * std_resid
	outlier_indices = np.where(np.abs(residuals) > threshold_sigma * std_resid)[0]
	
	# Initialize a boolean mask for flagged time samples
	flagged_mask = np.zeros(nsamp, dtype=bool)
	
	# For each outlier, flag it plus 'neighbor' samples on either side
	for idx in outlier_indices:
		start_idx = max(idx-neighbor, 0)
		end_idx = min(idx + neighbor, nsamp -1)
		flagged_mask[start_idx:end_idx +1] = True
		
	
	# If any samples are flagged, set the corresponding columns to NaN
	if np.any(flagged_mask):
		cols_to_flag = np.where(flagged_mask)[0]
		data_flagged[:, cols_to_flag] = np.nan
		n_flagged = cols_to_flag.size
		total_time_flagged = n_flagged * tsamp
		print(f'Flagged {n_flagged} time samples')
	else:
		print('No spikes detected above the threshold')
	
	# Optional diagnostic plot
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(x_seconds, time_avg, color='k', linewidth=1, label="Frequency-Averaged Power")
	ax.plot(x_seconds, y_fit, color='blue', linestyle='--', label='Spline Fit')
	
	thresh_min = y_fit - threshold_sigma * std_resid
	thresh_max = y_fit + threshold_sigma * std_resid
	ax.plot(x_seconds, thresh_max, 'r--', linewidth=1, label=f'$+{threshold_sigma}\sigma$')
	ax.plot(x_seconds, thresh_min, 'r--', linewidth=1, label=f'$-{threshold_sigma}\sigma$')
	if np.any(flagged_mask):
		ax.plot(x_seconds[outlier_indices], time_avg[outlier_indices], 'ro', markersize=4, label='Flagged Spikes')
		
	ax.set_xlim(x_seconds[0], x_seconds[-1])
	ax.set_ylim(np.min(y_fit_data) - 0.03, np.max(y_fit_data) + 0.03)
	ax.set_title(f'{hdr.source_name} Time-Averaged Spectrum (Stokes {stokes})')
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Mean Power")
	ax.legend()
	
	plt.savefig(f'{diag_path}.png', bbox_inches='tight', dpi=600)

	# plt.show()
	plt.close()
		
	return data_flagged
