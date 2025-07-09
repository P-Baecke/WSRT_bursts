'''
Plot the light curve and lomb scargle periodogram of the RT-1 WSRT observations

PB 2025
'''



# Import all needed modules
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from multiprocessing import Process

# Plot defaults
plt.switch_backend('wxAgg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['agg.path.chunksize'] = 10000

# Directory containing the light curves 
cache_dir = "/path/to/light/curves/"

plot_path = "/path/for/plots/"

# List of observation sources
source = ['ADLEO', 'EQPEG', 'WXUMA', 'PSV1', 'ILTJ1101']

# List all files in cache_dir
all_files = os.listdir(cache_dir)

periods = {
	'ADLEO': [2.227],
	'EQPEG': [1.061, 0.404],
	'WXUMA': [0.78, 3.5432],
	'PSV1': [0.00973669],
	'ILTJ1101': [0.08715]
	}

# Original sampling time (seconds)
tsamp = 0.016384

bin_intervals = [0.016384, 1, 10, 60, 360]


def bin_time(time, flux, old_tsamp, new_tsamp):
	'''
	Bin a 1d lightcurve to a coarse sampling
	
	Inputs:
		times (1D array)	: Time stamples of each sample (MJD)
		flux (1D array)	: Flux values at each time
		old_tsamp (float)	: Original sampling interval (seconds)
		new_tsamp (float)	: New sampling interval (seconds)
		
	Outputs:
		t_bin (1D array)	: Binned times (mean of each bin, in MJD)
		f_bin (1D array)	: Binned flux values (mean of each bin)		
	'''

	# determin how many samples are included in a bin
	bin_nsamp = int(round(new_tsamp / old_tsamp))
	if bin_nsamp < 1:
		raise ValueError(f'new_tsamp={new_tsamp}s must be >= old_tsamp={old_tsamp}s')
		
	# Total number of samples
	N = len(time)
	
	# COmpute number of full bins, include one extra if leftover exists
	num_full = N // bin_nsamp
	leftover = N % bin_nsamp
	num_bins = num_full + (1 if leftover else 0)
	
	# prepare output arrays
	t_bin = np.zeros(num_bins, dtype=float)
	f_bin = np.zeros(num_bins, dtype=float)
		
	# iterate over full bins
	for i in range(num_full):
		start = i * bin_nsamp
		end = start + bin_nsamp
		
		# mean time and flux in this bin
		t_bin[i] = np.nanmean(time[start:end])
		f_bin[i] = np.nanmean(flux[start:end])
		
	# Handle leftover as final bin
	if leftover:
		start = num_full * bin_nsamp
		t_bin[-1] = np.nanmean(time[start:N])
		f_bin[-1] = np.nanmean(flux[start:N])
		
	return t_bin, f_bin			

	
def process_source(src):


	# Find all csv files for this source (both Stokes I and V)
	files =  [fname for fname in all_files if fname.endswith('.csv') and (f'_{src}_') in fname]
	csv_files = sorted(files)
	
	
	# Read out the files and put into data arrays
	for stokes in ['I', 'V']:
		st_files = [fname for fname in csv_files if fname.endswith(f'_{src}_{stokes}.csv')]
		if not st_files:
			continue
			
		obs_list = [os.path.splitext(fnam)[0].split('_')[0] for fnam in st_files]
			
		times_list = []
		flux_list = []
		
		for fn in tqdm(st_files, desc=f'Loading {src}_{stokes}', leave=False):
			path = os.path.join(cache_dir, fn)
			data = np.loadtxt(path, delimiter=',', skiprows=1)
			times_list.append(data[:, 0])
			flux_list.append(data[:, 1])
		
		
		for dt in bin_intervals:
			
			print(f'Binning up to {dt}s')
			times_bin = []
			flux_bin = []
			
			for idx in range(len(times_list)):
				if dt == bin_intervals[0]:
					times_bin.append(times_list[idx])
					flux_bin.append(flux_list[idx])
				else:
					t_binned, f_binned = bin_time(times_list[idx], flux_list[idx], tsamp, dt)
					times_bin.append(t_binned)
					flux_bin.append(f_binned)
					
			# Per-epoch single lightcurve & LombScargle
			for idx, obs in tqdm(enumerate(obs_list), desc=f'Individual plots'):

				t = times_bin[idx]
				f = flux_bin[idx]
				
				
				# Lightcurve plot
				fig, ax = plt.subplots(figsize=(12, 6))
				ax.plot(t, f, color='k', linewidth=1)
				ax.set_xlabel('Time (MJD)')
				ax.set_ylabel('Flux density (arbitrary unit)')
				ax.set_title(f'{obs} {src} (Stokes {stokes}) - {dt}s binning')
			
				# Force full numbers on the x-axis
				ax.ticklabel_format(style='plain', useOffset=False, axis='x')
			
				
				if stokes == 'I':
					ax.set_ylim(0.8, 1.2)
				elif stokes == 'V':
					ax.set_ylim(0.5, 1.5)
				
				
				ax.set_xlim(t[0], t[-1])
			
				plt.savefig(f'{plot_path}LC_{dt}_{src}_{obs}_{stokes}.png', bbox_inches='tight', dpi=600)
			
				# plt.show()
				plt.close()

				# Mask out NaNs
				mask = ~np.isnan(f)
				t = t[mask]
				f = f[mask]
			
				# Default frequency maximum
				default_fmax = 5
				
				# Determine max freqeuncy based on known rotation periods
				if src in periods:
					# Use the shortest period to cover all known signals
					min_prot = min(periods[src])
					fmax = min(default_fmax, 3 / min_prot)
				else:
					fmax = default_fmax
				
			
				# Lomb-Scargle for this epoch
				window = np.ones_like(t)
				freq = np.linspace(0.01, fmax, 10000)
				ls = LombScargle(t, f)
				pwr = ls.power(freq)
				ls_w = LombScargle(t, window)
				pwr_w = ls_w.power(freq)

				f_fil = []
				p_fil = []
				pw_fil = []
				if src == 'ADLEO':
					for f, p, pw in zip(freq, pwr, pwr_w):
						if f<0.995 or f>1.005:
							f_fil.append(f)
							p_fil.append(p)
							pw_fil.append(pw)
				else:
					f_fil = freq
					p_fil = pwr
					pw_fil = pwr_w
					
				# Normalize to the maximum value for better visuals
				pwr_norm = p_fil / np.max(p_fil)
				pwr_w_norm = pw_fil / np.max(pw_fil)

				fig, ax = plt.subplots(figsize=(12, 6))
				ax.plot(f_fil, pwr_norm, label=f'Stokes {stokes}', color='k', linewidth=1)
				ax.plot(f_fil, -pwr_w_norm, label='Window function', color='orange', linewidth=1)
				ax.axhline(0, linestyle='--', linewidth='1', color='gray', alpha=0.75)
				ax.set_xlabel('Frequency (1/day)')
				ax.set_ylabel('Power')
				
				ax.set_title(f'{obs} {src} (Stokes {stokes}) - {dt}s binning')
			
				# Add known periodicities of sources				
				if src in periods:
					for i, Prot in enumerate(periods[src]):
						frot = 1.0 / Prot
						ax.axvline(frot, linestyle='--', linewidth='1', alpha=0.9, color='blue', label=f'P={Prot}d')
						ax.axvline(2*frot, linestyle='dotted', linewidth='1', alpha=0.5, color='blue')
						
			
					# Set xlim to 3.25x lower frot
					frot_min = 1.0 / max(periods[src])
					
				ax.set_xlim(min(freq), max(freq))
				ax.set_ylim(bottom=-1.1, top=1.1)
		
			
				ax.legend(loc='upper right')
				
				# plt.savefig(f'{plot_path}LS_{dt}_{src}_{obs}_{stokes}.png', bbox_inches='tight', dpi=600)
			
				# plt.show()
				plt.close()	
		
				
			
			times = np.hstack(times_bin)
			flux = np.hstack(flux_bin)
			
			# Plot the full lightcurve
			fig, ax = plt.subplots(figsize=(12, 6))
			ax.plot(times, flux, color='k')
			ax.set_xlabel('MJD')
			ax.set_ylabel(f'Flux density (arbitrary unit)')
			ax.set_title(f'Full Lightcurve {src} (Stokes {stokes}) - {dt}s binning')
			
			# Force full numbers on the x-axis
			ax.ticklabel_format(style='plain', useOffset=False, axis='x')
		
			ax.set_xlim(t[0], t[-1])	

			plt.savefig(f'{plot_path}LC_{dt}_{src}_full_{stokes}.png', bbox_inches='tight', dpi=600)
		
			# plt.show()
			plt.close()
		
		
			# mask out any NaNs in either array
			mask = (~np.isnan(flux))
			times = times[mask]
			flux  = flux[mask]
		
		
			# Default frequency maximum
			default_fmax = 5
			
			# Determine max freqeuncy based on known rotation periods
			if src in periods:
				# Use the shortest period to cover all known signals
				min_prot = min(periods[src])
				fmax = 3 / min_prot
			else:
				fmax = default_fmax
				
			
			# Define the window function: 1 at each observation time
			window = np.ones_like(times)
		
			# Compute Lomb-Scargle for data and window
			freq = np.linspace(0.01, fmax, 10000)
			print('Calculating LombScargle')
			ls_data = LombScargle(times, flux)
			power = ls_data.power(freq)
			print('Calculating Window Function')
			ls_window = LombScargle(times, window)
			power_win = ls_window.power(freq)
		
			f_fil = []
			p_fil = []
			pw_fil = []
			if src == 'ADLEO':
				for f, p, pw in zip(freq, power, power_win):
					if f<0.998 or f>1.0027:
						f_fil.append(f)
						p_fil.append(p)
						pw_fil.append(pw)
			else:
				f_fil = freq
				p_fil = power
				pw_fil = power_win
					
			
			# Normalize to the maximum value for better visuals
			pwr_norm = p_fil / np.max(p_fil)
			pwr_w_norm = pw_fil / np.max(pw_fil)
		
			
			# Plot the periodogram with mirrored window
			fig, ax = plt.subplots(figsize=(12, 6))
			ax.plot(f_fil, pwr_norm, label=f'Stokes {stokes}', color='k', linewidth=1)
			ax.plot(f_fil, -50*pwr_w_norm, label='Window function', color='orange', linewidth=1)
		
			# Zero line
			ax.axhline(0, linestyle='--', linewidth='1', color='gray', alpha=0.75)
		
			# Labels, legend, title
			ax.set_xlabel('Frequency (1/day)')
			ax.set_ylabel('Lomb-Scargle Power (arbitrary units)')
			ax.set_title(f'Lomb-Scargle & Window Function: {src} (Stokes {stokes}) - {dt}s binning')
		
			# Add known periodicities of sources				
			if src in periods:
				for i, Prot in enumerate(periods[src]):
					frot = 1.0 / Prot
					ax.axvline(frot, linestyle='--', linewidth='1', alpha=0.9, color='blue', label=f'P={Prot}d')
					ax.axvline(2*frot, linestyle='dotted', linewidth='1', alpha=0.5, color='blue')
					
					index = np.argmin(np.abs(freq - frot))
					p1 = power[index]
					fap_1 = ls_data.false_alarm_probability(p1)
					
					index = np.argmin(np.abs(freq - 2*frot))
					p2 = power[index]
					fap_2 = ls_data.false_alarm_probability(p2)
					
					print(f'\n{src} - {dt} - Stokes {stokes}\n')
					print(f'The FAP of {Prot} is: {fap_1}')
					print(f'The FAP 2x {Prot}: {fap_2}\n')
					
					
			ax.set_xlim(min(freq), max(freq))			
			ax.set_ylim(bottom=-1.1, top=1.1)

			ax.legend(loc='upper right')
	
			plt.savefig(f'{plot_path}LS_{dt}_{src}_full_{stokes}.png', bbox_inches='tight', dpi=600)
		
			# plt.show()
			plt.close()
		
	
if __name__ == "__main__":
	
	for src in source:

		print(f'\n{src}')
		p = Process(target=process_source, args=(src,))
		p.start()
		p.join()

