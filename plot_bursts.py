'''
This script contains all the function dealing with plotting as a supplement to radio_burst.py

PB 2025
'''

# Import all the necessary modules
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.signal import detrend, savgol_filter
import scipy.interpolate as interp
from astropy.timeseries import LombScargle

# plt.switch_backend('wxAgg')

# change the font
plt.rcParams['font.family'] = 'serif'

################################
#     PLOT DYNAMIC SPECTRUM    #
################################

def plot_dynspec(plot_data, hdr, start_sample, nsamp, tsamp, obs_name, pol_name, unit, counter, start_time, path, fname, freq_start_channel=0, freq_end_channel=None, savefig=False):
	'''
	This function is plotting dynamic spectra, and also calculates the lightcurves that are displayed on the side panels of the plot.
	Inputs:
		plot_data (2D array):	Array containing the dynamical spectrum time vs freq
		hdr (your.hdr)      :	Your header of the corresponding data  
		start_sample (int)  :	Starting sample of the data
		nsamp (int)         :	Number of samples that is being plotted
		tsamp (int)         :	Seconds per sample
		obs_name (str)      :	String indicating the observation set
		pol_name (str)      :	String indicating the Polarization 
		unit (str)          :	String indicating the time unit of the measurement
		counter (int)       :	Counter if several images of the same source in same obs
		start_time          :	Start time of the plot in UTC "YYYY-MM-DDTHH:MM:SS.sss"
		path (str)          :	Path where to save the plot
		fname (str)         :	Name under which to save the plot
		freq_start_channel  :	Starting frequency channel index (default 0)
		freq_end_channel    :	Ending fequency channel idnex (default None)
		savefig (boolean)   :	Boolean indicating whether the plot is being saved or not
	Returns:
		plot                : 	plt.show() showing the dynamical spectrum or saving it onto disk
	'''

	# Read in the object header to retrieve parameters
	source_name = hdr.source_name			# name of the source
	full_nchan = hdr.nchans				# Number of frequency channels
	freq_ch1 = hdr.fch1				# Frequencies of the first channel
	freq_offset = hdr.foff				# Frequency offset -> channel width (MHz)

	# If freq_end_channel is not provided, use the full range
	if freq_end_channel is None:
		freq_end_channel = full_nchan

	# Determine number of channels in the trimmed data
	nchan = freq_end_channel - freq_start_channel

	# Define the axis in physical units
	if unit == 's':
		time_axis = np.linspace(start_sample * tsamp, (start_sample + nsamp) * tsamp, nsamp)	# Time axis in s
	elif unit == 'min':
		time_axis = np.linspace(start_sample * tsamp / 60, (start_sample + nsamp) * tsamp / 60, nsamp)	# Time axis in min

	# Compute the full frequency axis based on the header, then trim it to the desired channels
	# The [::-1] reverses the axis so that higher frequencies are at the top
	full_freq_axis = 1e-3 * np.linspace(freq_ch1, freq_ch1 + freq_offset * full_nchan, full_nchan)		# Frequency axis in GHz

	freq_axis = full_freq_axis[freq_start_channel:freq_end_channel][::-1]


	# Initiate the plot and setup the layout
	fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8), width_ratios=[4, 1], height_ratios=[1, 4])

	# Calculate the 1st and 99th percentiles of the data, ignoring NaNs
	vmin = np.nanpercentile(plot_data, 5)
	vmax = np.nanpercentile(plot_data, 95)

	# For data that span positive and negative values (Stokes V), symmetry around zero
	if vmin < 0 and vmax > 0:
		bound = max(abs(vmin), abs(vmax))
		vmin, vmax = -bound, bound

	# Flip the data array to put the highest frequency channel at the top
	plot_data = plot_data[::-1, :]

	# Plot dynamic spectrum
	ax_dynspec = axes[1, 0]				# Initiate main panel ('bottom left')
	dynspec = ax_dynspec.imshow(plot_data, aspect='auto', origin='lower',
			extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
			cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='none')
	ax_dynspec.set_xlabel('Time ({})'.format(unit))
	ax_dynspec.set_ylabel('Frequency (GHz)')

	# Parse the start time
	start_time_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f")

	# Convert start time to Matplotlib's date format
	start_time_num = mdates.date2num(start_time_dt)

	# Convert relative time (in minutes) to absolute time
	def rel_to_abs(x):
		if unit=='min':
			return start_time_num + x / (24 * 60)  # Convert minutes to fraction of a day
		elif unit=='s':
			return start_time_num + x / (24 * 3600) # Convert seconds to fraction of a day
	# Convert absolute time back to relative minutes
	def abs_to_rel(x):
		if unit=='min':
			return (x - start_time_num) * 24 * 60  # Convert days back to minutes
		elif unit=='s':
			return (x - start_time_num) * 24 * 3600 # Convert days back to seconds

	raw_bandpass = np.nanmedian(plot_data, axis=1)

	# smooth_bandpass = savgol_filter(raw_bandpass, window_length=127, polyorder=6)

	# Time-average spectrum (top panel)
	time_average = np.nanmean(plot_data, axis=0)			# Average data over frequency
	ax_time_avg = axes[0, 0]				# Set it as top panel
	ax_time_avg.plot(time_axis, time_average, color='k', linewidth=0.8)
	ax_time_avg.tick_params(axis='x', labelbottom=False)

	# Add secondary x-axis
	secax = ax_time_avg.secondary_xaxis('top', functions=(rel_to_abs, abs_to_rel))
	secax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show only HH:MM:SS
	secax.set_xlabel('Absolute Time (UTC)')


	# Frequency-average spectrum (right panel)
	freq_average = np.nanmean(plot_data, axis=1)			# Average data over time
	ax_freq_avg = axes[1, 1]				# Set it as right panel
	ax_freq_avg.plot(freq_average, freq_axis, color='k', linewidth=0.8)
	# ax_freq_avg.plot(smooth_bandpass, freq_axis, color='r', linestyle='--', linewidth=0.8)
	# ax_freq_avg.set_xlim(-500, 500)
	ax_time_avg.tick_params(axis='y', labelleft='False')

	# More aestetics
	# plt.text(3, 6, obs_name)
	plt.subplots_adjust(wspace=0, hspace=0)			# Remove the space between the main plot and the side panels
	plt.colorbar(dynspec, label='Flux density (arbitrary unit)', ax=axes.ravel().tolist())	# Add colorbar
	fig.delaxes(axes[0, 1])					# Delete the top right panel since we don't need it
	plot_date = start_time_dt.strftime('%Y-%m-%d')
	plt.suptitle(f'{plot_date} - {obs_name}: Stokes {pol_name} of {source_name} : {counter}', fontsize=16, y=0.97)			# Set title

	if savefig:
		# Save the image
		file_path = '{}/{}.png'.format(path, fname)
		plt.savefig(file_path, bbox_inches='tight', dpi=600)
		
	# plt.show()
	plt.close()

################################
#   FOLDED PULSAR DYN SPEC     #
################################

def plot_folded_pulsar(folded_data, hdr, obs_name, pol_name, counter, freq_start_channel, freq_end_channel, nbin):
	'''
	Plot the folded spectrum of the calibration pulsars
	
	Parameters:
		folded_data (2D array)	: Folded profile data
		pol_name (str)	 	: Label for the used polarization
		nbin (int)		: Number of phase bins
		hdr (your.hdr)      :	Your header of the corresponding data  
	'''

	# Read in the object header to retrieve parameters
	source_name = hdr.source_name			# name of the source
	full_nchan = hdr.nchans				# Number of frequency channels
	freq_ch1 = hdr.fch1				# Frequencies of the first channel
	freq_offset = hdr.foff				# Frequency offset -> channel width (MHz)

	# If freq_end_channel is not provided, use the full range
	if freq_end_channel is None:
		freq_end_channel = full_nchan

	# Determine number of channels in the trimmed data
	nchan = freq_end_channel - freq_start_channel

	# Compute the full frequency axis based on the header, then trim it to the desired channels
	# The [::-1] reverses the axis so that higher frequencies are at the top
	full_freq_axis = 1e-3 * np.linspace(freq_ch1, freq_ch1 + freq_offset * full_nchan, full_nchan)		# Frequency axis in GHz

	freq_axis = full_freq_axis[freq_start_channel:freq_end_channel][::-1]

	# Computing the phase (x) axis
	phase = np.linspace(0, 1, nbin, endpoint=False)

	# Initiate the plot and setup the layout
	fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8), width_ratios=[4, 1], height_ratios=[1, 4])

	# Calculate the 1st and 99th percentiles of the data, ignoring NaNs
	vmin = np.nanpercentile(folded_data, 5)
	vmax = np.nanpercentile(folded_data, 95)

	# For data that span positiva and negative values (Stokes V), enformsymmetry around zero
	if vmin < 0 and vmax > 0:
		bound = max(abs(vmin), abs(vmax))
		vmin, vmax = -bound, bound

	# Flip the data array to put the highest frequency channel at the top
	folded_data = folded_data[::-1, :]

	# Plot dynamic spectrum
	ax_dynspec = axes[1, 0]				# Initiate main panel ('bottom left')
	dynspec = ax_dynspec.imshow(folded_data, aspect='auto', origin='lower',
			extent=[phase[0], phase[-1], freq_axis[0], freq_axis[-1]], 				cmap='Greys', vmin=vmin, vmax=vmax, interpolation='none')						# RdBu_r 
	ax_dynspec.set_xlabel('Pulse phase')
	ax_dynspec.set_ylabel('Frequency (GHz)')

	# Time-average spectrum (top panel)
	phase_average = np.nanmean(folded_data, axis=0)			# Average data over frequency
	ax_time_avg = axes[0, 0]				# Set it as top panel
	ax_time_avg.plot(phase, phase_average, color='k', linewidth=0.8)
	ax_time_avg.tick_params(axis='x', labelbottom=False)

	# Frequency-average spectrum (right panel)
	freq_average = np.nanmean(folded_data, axis=1)			# Average data over time
	ax_freq_avg = axes[1, 1]				# Set it as right panel
	ax_freq_avg.plot(freq_average, freq_axis, color='k', linewidth=0.8)
	ax_time_avg.tick_params(axis='y', labelleft='False')

	# More aestetics
	plt.subplots_adjust(wspace=0, hspace=0)			# Remove the space between the main plot and the side panels
	plt.colorbar(dynspec, label='Flux density (arbitrary unit)', ax=axes.ravel().tolist())	# Add colorbar
	fig.delaxes(axes[0, 1])					# Delete the top right panel since we don't need it
	
	# Parse the start time
	start_time_dt = datetime.strptime(hdr.tstart_utc, "%Y-%m-%dT%H:%M:%S.%f")

	plot_date = start_time_dt.strftime('%Y-%m-%d')
	plt.suptitle(f'{plot_date} - {obs_name}: Stokes {pol_name} of {source_name} : {counter}', fontsize=16, y=0.97)			# Set title

	# Save and show the image
	path = '/home/paulbaecke/Research_Project/plots/7/'
	file_name = '{}_{}_{}_{}'.format(obs_name, source_name, pol_name, counter)
	file_path = '{}{}.pdf'.format(path, file_name)
	# plt.savefig(file_path, bbox_inches='tight')
	file_path = '{}{}.png'.format(path, file_name)
	# plt.savefig(file_path, bbox_inches='tight')

	plt.show()
	plt.close()
	
	
################################
#    BANDPASS COMPARISON PLOT  #
################################

def plot_bandpass(hdr, I_bandpass, I_bandpass_SG, I_bandpass_SPL, num_ticks=10):
	'''
	Plots bandpass correction curves using header information and various bandpass arrays.
	
	Parameters:
		hdr (your object)	: Header of the corresponding data file
		I_bandpass (1D array)	: Bandpass of standard corrected data
		I_bandpass_SG (1D arr)	: Bandpass of savgol corrected data
		I_bandpass_SPL(1D arr)	: Bandpass of spline corrected data
		num_ticks (int, opt)	: Number of tick labels to display along the frequency x-axis (default is 10)
		
	Returns:
		None, this function displays the plot and closes the figure
	'''
	
	# Read in the object header to retrieve parameters
	full_nchan = hdr.nchans				# Number of frequency channels
	freq_ch1 = hdr.fch1				# Frequencies of the first channel
	freq_offset = hdr.foff				# Frequency offset -> channel width (MHz)

	# Compute the full frequency axis based on the header, then trim it to the desired channels
	# The [::-1] reverses the axis so that higher frequencies are at the top
	full_freq_axis = 1e-3 * np.linspace(freq_ch1, freq_ch1 + freq_offset * full_nchan, full_nchan)		# Frequency axis in GHz
	
	# Compute the frequency values corresponding to the x-axis ticks
	freq_labels = full_freq_axis[::-1]  # Reverse to match the original order

	# Define the number of tick labels you want to display
	num_ticks = 10  # Adjust as needed
	tick_indices = np.linspace(0, len(freq_labels) - 1, num_ticks, dtype=int)  # Select evenly spaced indices
	tick_positions = tick_indices  # The original x-axis positions
	tick_labels = [f"{freq_labels[i]:.2f}" for i in tick_indices]  # Format labels to 2 decimal places
	
	# Plot the actual bandpass and the fitted bandpass
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(I_bandpass, label="Old Bandpass", linestyle='-', c='k')
	ax.plot(I_bandpass_SG, label="SG Bandpass", linestyle='--', c='red')
	ax.plot(I_bandpass_SPL, label="SPL Bandpass", linestyle='--', c='green')
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Intensity")
	
	ax.set_xlim(0, full_nchan-1)
		
	# Set new x-axis labels
	ax.set_xticks(tick_positions)
	ax.set_xticklabels(tick_labels)

	ax.legend()
	ax.grid()
	plt.title('Bandpass Fitting: Savgol vs Spline')
	plt.show()
	plt.close()


################################
#          TIME AVERAGE        #
################################

def plot_freq_average_spectrum(data, hdr, stokes):
	'''
	Compute and plot the frequency-averaged spectrum for a given Stokes parameter.

	Inputs:
		data (2D array)     : Dynamic spectrum for the selected Stokes parameter (nchan x nsamp)
		hdr (object)        : Header object containing metadata (must include hdr.source_name)
		stokes (str)        : Stokes parameter identifier ('I', 'V', etc.)

	Outputs:
		spectrum (1D array) : Time-averaged spectrum across all time samples
	'''

	# Compute the time-averaged spectrum
	spectrum = np.nanmean(data, axis=0)
	
	# Time axis (if available from header, insert here)
	freqs = np.arange(data.shape[1])  # Placeholder for frequency axis

	# Plotting
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(freqs, spectrum, color='k')
	ax.set_xlabel('Time sample')
	ax.set_ylabel('Mean Power')
    
	ax.axhline(y=np.nanmean(spectrum), linestyle='-', color='r')
	ax.axhline(y=np.nanmean(spectrum) - 3*np.nanstd(spectrum), linestyle=':', color='r')
	ax.axhline(y=np.nanmean(spectrum) + 3*np.nanstd(spectrum), linestyle=':', color='r')
    
	ax.set_title(f'{hdr.source_name} Frequency-Averaged Spectrum (Stokes {stokes})')
	plt.show()
	plt.close()
	
    

################################
#       HISTOGRAM OF DATA      #
################################

def plot_hist(plot_data, title):
	'''
	This function is plotting dynamic spectra, and also calculates the lightcurves that are displayed on
	the side panels of the plot.
	Inputs:
		plot_data (2D array)	: Array containing the dynamical spectrum time vs freq
		title (str)		: Your header of the corresponding data  
	Returns:
		plot                : 	plt.show() showing the dynamical spectrum or saving it onto disk
	'''
	
	
	# Flatten the array for 1D histogram plotting
	data_flat = plot_data.ravel()

	# Calculate mean, median and standard deviation	
	mean = round(np.nanmean(plot_data), 2)
	median = round(np.nanmedian(plot_data), 2)
	std = round(np.nanstd(plot_data), 2)
	
	# Plot the histogram
	fig, ax = plt.subplots(figsize=(12, 6))
	
	ax.hist(data_flat, bins=1000, alpha=0.7, label='Entire Data', color='k')
	
	ax.axvline(x=mean, linestyle='-', color='k', label=f'Mean {mean}')
	ax.axvline(x=median, linestyle='--', color='k', label=f'Median {median}')
	
	# Set x axis limit to 5 sigma around the mean
	ax.set_xlim(mean - 5 * std, mean + 5*std)
	
	ax.set_xlabel('Data Values')
	ax.set_ylabel('Number')
	ax.set_title(title)
	ax.legend()
	plt.show()
	plt.close()
	
	
################################
#      PERIODOGRAM OF DATA     #
################################

def plot_lombscargle_periodogram(data, hdr, tsamp, stokes, max_period=30):
	'''
	Compute and plot a Lomb-Scargle periodogram for a given Stokes parameter signal,
	handling NaN values and converting frequency to period.

	Inputs:
		data (2D array)     : Dynamic spectrum for the selected Stokes parameter (nchan x nsamp)
		hdr (object)        : Header object containing metadata (must include 		hdr.source_name)
		tsamp (float)       : Sampling time in seconds per time sample
		stokes (str)        : Stokes parameter identifier ('I', 'V', etc.)
		max_period (float)  : Maximum period in seconds to display in the plot
	'''	

	# Time vector in seconds
	t = np.arange(data.shape[1]) * tsamp

	# Average over frequency axis to get a single time series
	series = np.nanmean(data, axis=0)
	mask = np.isfinite(series)
	t_clean = t[mask]
	y_clean = series[mask]

	# Compute Lomb-Scargle periodogram
	ls = LombScargle(t_clean, y_clean)
	freq, power = ls.autopower()

	# Plotting
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(1 / freq, power)
	ax.set_xlim(0, max_period)
	ax.set_xlabel('Period (s)')
	ax.set_ylabel('Power')
	ax.set_title(f'{hdr.source_name} Periodicity in Stokes {stokes}')
	plt.show()
	plt.close()
