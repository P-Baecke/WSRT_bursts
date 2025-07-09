'''
This script does analysis calculations of the noise reduced single dish WSRT data.

It reads out all the processed data and calculates and puts out:
 - overall flagging statistics
 - flagging rates per category
 - dynamic range assessment
 - Noise levels in Stokes I and V
 - light curve for single source
 - Plots periodicity versus windowing function using Lomb-Scargle

PB 2025
'''

import os
import your
import csv
import numpy as np
from astropy.timeseries import LombScargle
import astropy.time as atime
import matplotlib.pyplot as plt
from tqdm import tqdm
from file_handling import assign_calibration_by_order, get_data
from plot_bursts import plot_freq_average_spectrum

plt.switch_backend('wxAgg')

# change the font
plt.rcParams['font.family'] = 'serif'

# Define observation names
epochs = ['pex006', 'pex007', 'pex008', 'pex009', 'pex010', 'pex011', 'pex012', 'pex013', 'pex014', 'pex015', 'pex016', 'pex017', 'pex018']

# Define the base path of the processed data
base_dir = '/data/minoss-vdb/wsrt_bursts/processed/'

cache_dir = '/home/paulbaecke/Research_Project/analysis_cache/mf/'

os.makedirs(cache_dir, exist_ok=True)


# Output report file
report_path = 'analysis_summary.txt'

# Initialize accumulators
epoch_flag_fracs = {}
source_flag_fracs = {}
subband_flag_fracs = {sb: [] for sb in range(8)}
overall_flags_I = []
overall_flags_V = []
noise_I = []
noise_V = []
lightcurves = {}

# Helper functions
def calc_flag_fraction(data):
	'''
	Calculate the fraction of flagged data (NaNs) in the input array.
	
	Parameters:
		data (2D array)	: Input dynamic spectrum
	Outputs:
		frac (float)		: Fraction of the NaN values.
	'''
	
	size = data.size
	
	flagged = np.count_nonzero(np.isnan(data))
	
	frac = flagged / size
	
	return frac
	
	

	
	
for obs in tqdm(epochs, desc='Epochs'):

	# initialize arrays
	tarr_adleo = []
	tarr_eqpeg = []
	tarr_wxuma = []
	tarr_psv1 = []
	tarr_iltj1101 = []
	lc_adleo_I = []
	lc_eqpeg_I = []
	lc_wxuma_I = []
	lc_psv1_I = []
	lc_iltj1101_I = []
	lc_adleo_V = []
	lc_eqpeg_V = []
	lc_wxuma_V = []
	lc_psv1_V = []
	lc_iltj1101_V = []
	
	file_path = os.path.join(base_dir, obs, '02_ManFlag_FIL/')

	files = [file for file in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, file))]
	
	# Sort the files alphabetically (in order of observation)
	files = sorted(files)
	
	calibration_map, source_files, pulsar_files, source_counts = assign_calibration_by_order(files, file_path)
	
	# Container for this epoch
	print(f'------\n{file_path}\n')
	
	for idx, fname in enumerate(tqdm(source_files, desc=f'{obs}', leave=False)):
		
		# Create a Your object
		your_object = your.Your(fname)
		
		# Read out the header of the your file
		hdr = your_object.your_header
		
		source_name = hdr.source_name		# Name of the source
		src_name = source_name.strip().upper()
		tsamp = hdr.tsamp			# Sample time in seconds
		nsamp = hdr.nspectra			# Number of time samples
		tstart_mjd = hdr.tstart			# Starting time in UTC
		
		# Print out the source that's being analysed 
		print('{}/{} - Source: {}'.format(idx+1, len(source_files), source_name))
		
		data = get_data(your_object, nstart=0, nsamp=nsamp, npoln=1, keys='infer')
		
		flag_frac = calc_flag_fraction(data)
		
		if fname.endswith("_I.fil"):
			stokes = 'I'
			overall_flags_I.append(flag_frac)
		elif fname.endswith("_V.fil"):
			stokes = 'V'
			overall_flags_V.append(flag_frac)
		
		source_flag_fracs.setdefault(src_name, []).append(flag_frac)
		
		print('Flagging fraction: {}'.format(round(flag_frac, 6)))
		
		## per subband
		nchans = hdr.nchans
		sbw = nchans // 8
		
		for sb in range(8):
			sub = data[sb*sbw:(sb+1)*sbw, :]
			subfrac = calc_flag_fraction(sub)
			subband_flag_fracs[sb].append(subfrac)
			
		
		
		# Calculate the time_array
		time_array = tstart_mjd + np.arange(nsamp) * (tsamp / 86400.0)
		
		# Calculate the lightcurve
		lightcurve = plot_freq_average_spectrum(data, hdr, stokes)
		
		# Array selection logic
		if src_name == 'ADLEO':
			if stokes == 'I':
				tarr_adleo.append(time_array)
				lc_adleo_I.append(lightcurve)		
			elif stokes == 'V':
				lc_adleo_V.append(lightcurve)
		elif src_name == 'EQPEG':
			if stokes == 'I':
				tarr_eqpeg.append(time_array)
				lc_eqpeg_I.append(lightcurve)
			elif stokes == 'V':
				lc_eqpeg_V.append(lightcurve)
		elif src_name == 'WXUMA':
			if stokes == 'I':
				tarr_wxuma.append(time_array)
				lc_wxuma_I.append(lightcurve)
			elif stokes == 'V':
				lc_wxuma_V.append(lightcurve)
		elif src_name == 'PSV1':
			if stokes == 'I':
				tarr_psv1.append(time_array)
				lc_psv1_I.append(lightcurve)
			elif stokes == 'V':
				lc_psv1_V.append(lightcurve)
		elif src_name == 'ILTJ1101':
			if stokes == 'I':
				tarr_iltj1101.append(time_array)
				lc_iltj1101_I.append(lightcurve)
			elif stokes == 'V':
				lc_iltj1101_V.append(lightcurve)
				


	# Only stack & save if data exists
	if tarr_adleo:
		tarr = np.hstack(tarr_adleo)
		lcI  = np.hstack(lc_adleo_I)
		np.savetxt(f"{cache_dir}{obs}_ADLEO_I.csv",
			np.vstack([tarr, lcI]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")
	if tarr_adleo and lc_adleo_V:
		tarr = np.hstack(tarr_adleo)
		lcV  = np.hstack(lc_adleo_V)
		np.savetxt(f"{cache_dir}{obs}_ADLEO_V.csv",
			np.vstack([tarr, lcV]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")

	if tarr_eqpeg:
		tarr = np.hstack(tarr_eqpeg)
		lcI  = np.hstack(lc_eqpeg_I)
		np.savetxt(f"{cache_dir}{obs}_EQPEG_I.csv",
			np.vstack([tarr, lcI]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")
	if tarr_eqpeg and lc_eqpeg_V:
		tarr = np.hstack(tarr_eqpeg)
		lcV  = np.hstack(lc_eqpeg_V)
		np.savetxt(f"{cache_dir}{obs}_EQPEG_V.csv",
			np.vstack([tarr, lcV]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")

	if tarr_wxuma:
		tarr = np.hstack(tarr_wxuma)
		lcI  = np.hstack(lc_wxuma_I)
		np.savetxt(f"{cache_dir}{obs}_WXUMA_I.csv",
			np.vstack([tarr, lcI]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")
	if tarr_wxuma and lc_wxuma_V:
		tarr = np.hstack(tarr_wxuma)
		lcV  = np.hstack(lc_wxuma_V)
		np.savetxt(f"{cache_dir}{obs}_WXUMA_V.csv",
			np.vstack([tarr, lcV]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")

	if tarr_psv1:
		tarr = np.hstack(tarr_psv1)
		lcI  = np.hstack(lc_psv1_I)
		np.savetxt(f"{cache_dir}{obs}_PSV1_I.csv",
			np.vstack([tarr, lcI]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")
	if tarr_psv1 and lc_psv1_V:
		tarr = np.hstack(tarr_psv1)
		lcV  = np.hstack(lc_psv1_V)
		np.savetxt(f"{cache_dir}{obs}_PSV1_V.csv",
			np.vstack([tarr, lcV]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")

	if tarr_iltj1101:
		tarr = np.hstack(tarr_iltj1101)
		lcI  = np.hstack(lc_iltj1101_I)
		np.savetxt(f"{cache_dir}{obs}_ILTJ1101_I.csv",
			np.vstack([tarr, lcI]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")
	if tarr_iltj1101 and lc_iltj1101_V:
		tarr = np.hstack(tarr_iltj1101)
		lcV  = np.hstack(lc_iltj1101_V)
		np.savetxt(f"{cache_dir}{obs}_ILTJ1101_V.csv",
			np.vstack([tarr, lcV]).T,
			delimiter=",",
			header="time_mjd,flux",
			comments="")

with open('flagging_summary.csv', 'w', newline='') as f:
	w = csv.writer(f)
	w.writerow(['category', 'key', 'value'])
	
	# per-epoch
	for obs, frac in epoch_flag_fracs.items():
		w.writerow(['epoch', obs, f'{round(frac, 6)}'])
		
	# per-source
	for src, fracs in source_flag_fracs.items():
		w.writerow(['source', src, f'{round(np.mean(fracs), 6)}'])
		
	# per subband
	for sb, fracs in subband_flag_fracs.items():
		w.writerow(['subband', sb, f'{round(np.mean(fracs), 6)}'])
		
	if overall_flags_I:
		w.writerow(['overall_I', 'mean_flag_frac', f'{round(np.mean(overall_flags_I), 6)}'])
		w.writerow(['overall_I', 'cum_flag_frac', f'{round(np.sum(overall_flags_I), 6)}'])
		w.writerow(['overall_I', 'n_files', len(overall_flags_I)])

	if overall_flags_V:
		w.writerow(['overall_V', 'mean_flag_frac', f'{round(np.mean(overall_flags_V), 6)}'])
		w.writerow(['overall_V', 'cum_flag_frac', f'{round(np.sum(overall_flags_V), 6)}'])
		w.writerow(['overall_V', 'n_files', len(overall_flags_V)])

	# Combined (if desired)
	if overall_flags_I and overall_flags_V:
		all_flags = overall_flags_I + overall_flags_V
		w.writerow(['overall', 'mean_flag_frac', f'{round(np.mean(all_flags), 6)}'])
		w.writerow(['overall', 'cum_flag_frac', f'{round(np.sum(all_flags), 6)}'])
		w.writerow(['overall', 'n_files', len(all_flags)])



