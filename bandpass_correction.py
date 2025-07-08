'''
This script contains all the function dealing with bandpass correction as a supplement to radio_burst.py

PB 2025
'''

# import all necessary modules
import numpy as np
from scipy.signal import detrend, savgol_filter
import scipy.interpolate as interp
import matplotlib.pyplot as plt


################################
# BASIC SUBBAND NORMALIZATION  #
################################

def correct_bandpass(data, n_subbands=8, return_extra=False):
	"""
	Bandpass corrects a 2D array by normalizing each frequency subband.

	Parameters
	----------
	data : 2D numpy array
        	Data to be normalized. Frequency on the y-axis (nchans) and time samples on the x-axis.
	n_subbands : int, optional
	        Number of frequency subbands to divide the data into (default is 8).
	return_extra : bool, optional
	        If True, also returns the mean and std of each subband (default is False).

	Returns
	-------
	data_bpc : 2D numpy array
	        Bandpass-corrected data where (data - mean) / std per subband.
	subband_mean, subband_std : 1D numpy arrays (only if return_extra=True)
	        Mean and standard deviation of each frequency subband.
	"""

	print('Applying bandpass correction using', n_subbands, 'frequency subbands.')

	nchans, nsamp = data.shape  # Get frequency channels (y-axis) and time samples (x-axis)

	if nchans % n_subbands != 0:
	        raise ValueError("Number of channels must be evenly divisible by the number of subbands.")

	subband_size = nchans // n_subbands  # Number of channels per subband
	complete_mean = np.zeros(nchans)
	complete_std = np.zeros(nchans)


	for i in range(n_subbands):
		subband_start = i * subband_size
		subband_end = (i + 1) * subband_size

		# Compute mean and standard deviation across the entire time axis for each subband
		mean_val = np.nanmean(data[subband_start:subband_end, :], axis=1)
		std_val = np.nanstd(data[subband_start:subband_end, :], axis=1)

		complete_mean[subband_start:subband_end] = mean_val
		complete_std[subband_start:subband_end] = std_val

		subband_mean = np.nanmean(data[subband_start:subband_end, :])
		subband_std = np.nanstd(data[subband_start:subband_end, :])

		# Normalize the subband
		data[subband_start:subband_end, :] = (data[subband_start:subband_end,  :] - subband_mean) / subband_std


	# Mitigate division by zero
	complete_std[complete_std == 0] = 1.0

	if return_extra:
		return data, complete_mean, complete_std

	return data

################################
#    SAVGOL FIT NORMALIZATION  #
################################

def correct_bandpass_savgol(data, n_subbands=8, window_length=127, polyorder=6, return_bandpass=True):
	'''
	This function applies a bandpass correction by computing the time-averaged (median) spectrum per channel, smoothing it using a Savitky-Golay filter, and normalizing the data.

	Inputs:
		data (2D array)	: The 2D array containing the data to be corrected
		n_subbands (int)	: Number of frequency subbands to divide the data into (8)
		window_length (int)	: Length of the filter window (must be odd)
		polyorder (int)	: Polynomial order for the filter (must be less than the window length
		return_bandpass (bool)	: If True, also return the smoothed bandpass curve
	Returns:
		data_bpc (2D array)	: 2D array containing the corrected data
		bandpass (1D array)	: The smoothed bandpass curve (if return_bandpass=True)	
	'''
	nchans, nsamp = data.shape
	if nchans % n_subbands != 0:
		raise ValueError('Number of channels must be evenly dicisible by the number of subbands')

	subband_size = nchans // n_subbands
	smooth_bandpass = np.zeros(nchans)

	for i in range(n_subbands):
		subband_start = i * subband_size
		subband_end = (i+1) * subband_size


		# Compute the median spectrum along the time axis per channel
		median_spectrum = np.nanmedian(data[subband_start:subband_end, :], axis=1)

		# Identify channels where the median is NaN (i.e. the channel has all NaN values)
		invalid_channels = np.isnan(median_spectrum)

		channel_indices = np.arange(subband_start, subband_end)

		# Interpolate missing values
		if np.any(invalid_channels):
			# Identify valid channels (with non-NaN median)
			valid_channels = ~invalid_channels

			if np.sum(valid_channels) > 0:
				# Interpolate to fill in the missing median vlaues for channels with all NaNs
				median_spectrum[invalid_channels] = np.interp(channel_indices[invalid_channels], channel_indices[valid_channels], median_spectrum[valid_channels])


			else:
				# If no valid channel exists, set all median values to 1 to avoid division by zero
				median_spectrum[:] = 1.0

		# Smooth the median spectrum with a Savitzky-Golay filter
		smooth_bandpass[subband_start:subband_end] = savgol_filter(median_spectrum, window_length, polyorder)
		# smooth_bandpass[subband_start:subband_end] = savgol_filter(median_spectrum, window_length, polyorder)

		# Mitigate division by zero
		smooth_bandpass[smooth_bandpass == 0] = 1.0

		# Normalize the data, each channel is divided by its smoothed bandpass value
		data[subband_start:subband_end, :] /= smooth_bandpass[subband_start:subband_end, None]

	if return_bandpass:
		return data, smooth_bandpass
	return data

################################
# SMOOTH SPLINE NORMALIZATION  #
################################

def correct_bandpass_spline(data, diag_path, n_subbands=8, smooth_param=1, return_bandpass=True):
	'''
	Applies a bandpass correction per subband using a smoothing spline.
	
	The function computes the time-averaged (median) spectrum per channel, fills in missin (NaN) values by linear interpolation, a smoothing spline to the edian spectrum sunig scipy's make_smoothing_spline, and the normalizes each channel by its smoothed bandpass value.
	
	Parameters:
		data (2D array)	: The input data to correct
		diag_path (str)	: Directory path where to save the diagnostic plot
		n_subbands (int)	: Number of frequency subbands to divide the data into
		smooth_param (float)	: Smoothing parameter for the spline (controls smoothness) Lower values yield a closer fit to the data.
		
	Returns:
		data_bpc (2D array)	: The bandpass corrected data
		bandpass (1D array)	: optional: the smoothes bandpass curve, if return_bandpass=True
	'''
	
	nchans, nsamp = data.shape
	
	# Check that the number of channels is evenly divisible into subbands
	if nchans % n_subbands !=0:
		raise ValueError('Number of channels must be evenly divisible by the number of subbands.')
		
	subband_size = nchans // n_subbands		# Number of channels per subband
	smooth_bandpass = np.zeros(nchans)		# Initiate array
	
	
	# full_median = np.nanmedian(data_masked, axis=1)	# Median of the entire spectrum
	full_median = np.nanmedian(data, axis=1)		# Median of the entire spectrum
	
	# Process each subband separately
	for i in range(n_subbands):
		subband_start = i * subband_size
		subband_end = (i+1) * subband_size
		
		# Compute the median spectrum for each channel in this subband
		median_spectrum = full_median[subband_start:subband_end]
		
		# Create an array of channel indices for the current subband
		channel_indices = np.arange(subband_start, subband_end)

		# Identify channels with naN median values ( e.g., due to flagged data)
		invalid_channels = np.isnan(median_spectrum)
		
		# interpolate over naN values if any exists
		if np.any(invalid_channels):
			valid_channels = ~invalid_channels
			if np.sum(valid_channels) > 0:
				# Replace NaNs with linear interplation from valid channels
				median_spectrum[invalid_channels] = np.interp(
					channel_indices[invalid_channels], 
					channel_indices[valid_channels],
					median_spectrum[valid_channels]
				)
			else:
				# If no valid values exist, set the entire subband to ones to avoid division by zer0
				median_spectrum[:] = 1.0
				
		# Fit a smoothing spline to the median spectrum 
		# Note: make_smoothing_spline does not support NaN values, but they are already interpolated
		spline = interp.make_smoothing_spline(channel_indices, median_spectrum, lam=smooth_param)
		
		smoothed = spline(channel_indices)

		# Prevent division by zero by ensuring smoothed values are nonzero
		smoothed[smoothed == 0] = 1.0
		
		# Store the smoothed bandpass values for the current subband
		smooth_bandpass[subband_start:subband_end] = smoothed
		
		
		# Normalize the subband data by dividing each channel by its corresponding smoothed value
		data[subband_start:subband_end, :] = (data[subband_start:subband_end, :] /smoothed[:, None])
	
	# Control plot	
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(np.arange(len(full_median)), full_median, '-', color='k', label='Data median')
	ax.plot(np.arange(len(full_median)), smooth_bandpass, '-', color='r', label='Spline fit')
	ax.set_xlabel('Frequency channel')
	ax.set_ylabel('Median power')
	ax.set_xlim(0, len(full_median))
	plt.legend()
	
	plt.savefig(f'{diag_path}.png', bbox_inches='tight', dpi=600)
	# plt.show()
	plt.close()
		
	if return_bandpass:
		return data, smooth_bandpass
	return data

	
################################
#  TIME NORMALIZATION SPLINE   #
################################


def time_spline(data, tsamp, diag_path, n_subbands=8, smooth_param=250, return_spline=True):
	'''
	Perform a rough spline fit over the time axis of a dynamic spectrum and normalize out the slow variations
	
	Inputs:
		data (2D array)	: The input data, a dynamic spectrum (nchan, nsamp)
		tsamp (float)		: Time in s per time sample
		diag_path (str)	: Path where to save the diagnostic plot
		n_subbands (int)	: Number of frequency subbands to divide the data into
		smooth_param (float)	: Smoothing parameter for the spline (controls smoothness). Lower values yield a closer fit to the data
		return_spline (bool)	: Boolean whether to return the fitted function or not.
		
	Outputs:
		data_norm (2D array)	: The input data normalized by dividing out the time-spline
		spline (1D array)	: The fitted spline values at each time sample
	'''
	
	
	nchans, nsamp = data.shape
	
	# Check that the number of channels is evenly divisible into subbands
	if nchans % n_subbands !=0:
		raise ValueError('Number of channels must be evenly divisible by the number of subbands.')
		
	subband_size = nchans // n_subbands		# Number of channels per subband
		
	# build a time vector in seconds
	t = np.arange(nsamp) * tsamp
	
	# Calculating overall mean 
	time_mean_full = np.nanmean(data, axis=0)
	
	# Storing each subbands spline
	spline_curves = []
	
	# Iterate through each subband
	for i in range(n_subbands):
		start = i * subband_size
		end = (i+1) * subband_size
		sub_data = data[start:end, :]		# Extract the data for one subband
	
		# Compute the median across channels for each time sample
		time_mean = np.nanmean(sub_data, axis=0)
	
		# counting number of NaN's
		valid = ~np.isfinite(time_mean)
		
		if np.count_nonzero(~valid) < 10:
			# Too few valid data points to fit spline
			spline_curves.append(None)
			print(f"Subband {i+1} skipped (only {nsamp-np.count_nonzero(valid)} valid time samples).")
			continue
			
		# If there are any NaNs, do a quick linear fill
		if np.any(valid):
			# Define an interp over the finite points
			fin = np.isfinite(time_mean)
			f_interp = interp.interp1d(t[fin], time_mean[fin],
					kind='linear',
					bounds_error=False,
					fill_value="extrapolate")
				

			# Replace the NaNss
			time_mean = f_interp(t)
	
		# Fit the smoothing spline
		spline = interp.make_smoothing_spline(t, time_mean, lam=smooth_param)
	
		# Evaluate the spline at each timestamp
		spline_curve = spline(t) 
	
		# Avoid division by zero
		spline_curve[spline_curve == 0] = 1.0
	
		
		# Save spline for optional return and plotting
		spline_curves.append(spline_curve)
	
		# Normalize each column of the data by the corresponding spline value
		data[start:end, :] /= spline_curve[None, :]
	
	
	# Only keep spline arrays that are not None
	valid_splines = [sc for sc in spline_curves if sc is not None]

	# Convert to a 2D array (n_valid_subbands x nsamp)
	spline_array = np.array(valid_splines)

	# Compute mean across subbands
	mean_spline = np.mean(spline_array, axis=0)

	# For clean plotting
	fig, ax = plt.subplots(figsize=(12, 6))
	
	ax.plot(t, time_mean_full, '-', color='k', label='Data mean')
	ax.plot(t, mean_spline, '-', color='r', label='Spline fit')
		
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Mean power')
	ax.set_xlim(t[0], t[-1])
	ax.set_title('Spline fit over time')
	
	plt.legend()
		
	plt.savefig(f'{diag_path}.png', bbox_inches='tight', dpi=600)
	# plt.show()
	plt.close()
		
	if return_spline:
		return data, spline_curves
	return data
	
