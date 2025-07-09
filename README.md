# WSRT_bursts

The repository contains the code used for a feasibility study on single-dish WSRT observations.

It contains a noise removal pipeline with additional functions.

Pipeline description:

01_calc_stokes_man_flag.py
  Run in terminal with all the paths specified
  Reads in full Stokes filterbank files
  Form the Stokes parameters I and V
  Plot dynamic spectra for raw data
  Flag the edges of the subbands
  Flag entire subbands
  Flag startup irregularities
  Saves the files as filterbank files

AOFlagger #1:
  In terminal:
  aoflagger -v -strategy /path/to/wsrt-rt2-0.lua /path/to/data/*
  This automatically flags common and mid-intense RFI
  Be aware that this overwrites your files

AOFlagger #2
  In terminal:
  aoflagger -v -strategy /path/to/wsrt-rt2-1.lua /path/to/data/*
  This automatically flags RFI, especially aggressive towards sharp RFI
  Be aware that this overwrites your files

02_spline.py
  Reads in the flagged filterbank files
  Plots the dynamic spectra
  Normalises the bandpass using a subband-wise spline fit
  Removing obvious outliers (flagging)
  Plots the dynamic spectra after bandpass correction
  Saves the files as filterbank files

AOFlagger #3
  In terminal:
  aoflagger -v -strategy /path/to/wsrt-rt2-2.lua /path/to/data/*
  This automatically flags blob-like RFI
  Be aware that this overwrites your files

03_stitch.py
  Reads in the filterbank files
  Removes sharp RFI spikes on the time axis
  Adjust for time variations using a spline fit
  Plot and save the dynamic spectra

analysis.py
  Reads out the processed data
  Calculates the overall flagging statistics
  Saves the light curves of the sources as .csv files

lomb-scargle.py
  Bins up the light curves to 1s, 10s, 60s, and 360s
  Plots the light curves
  Performs a Lomb-Scargle periodicity search and plots the power and window function

  
  
