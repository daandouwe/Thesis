Alice fMRI ROI Timecourses
Jonathan Brennan <jobrenn@umich.edu>
ORCID: 0000-0002-3639-350X


These files contain fMRI time-courses from 6 regions of interest recorded from participants at the Cornell MRI center while participants listened to 12 min of Chapter 1 of Alice in Wonderland.

These files are the bases of the results reported in:

Brennan, J. R., Stabler, E. P., Van Wagenen, S. E., Luh, W.-M., & Hale, J. T. (2016). Abstract linguistic structure correlates with temporal activity during naturalistic comprehension. Brain Lang, 157-158, 81–94.

ROI definitions and stimulus-derived time-series are described in that publication

Filename descriptions

6mm/*  	- Timeseries from 6 regions (columns) and 362 samples (rows) from 6mm radius ROIs
10mm/* 	- Timeseries from 6 regions (columns) and 362 samples (rows) from 10mm radius ROIs

	Column labels are:
	LATL RATL LPTL LIPL LPreM LIFG
	
	Note: The ten initial fMRI volumes were discarded from the raw data

s##-rate.txt 	- timeseries for the "word rate" predictor derived from the aud stimulus
s##-sndpwr.txt 	- timeseries for the "sound power" predictor derived from aud stim
s##-rp_*.txt 	- per-subject movement parameters estimated with SPM8

sept18-generic-predictors... - other linguistic predictors used in data analysis

	column labels are:
	rate_conv frq_conv_orth break_conv_orth tdp_conv_orth tdx_conv_orth bup_conv_orth bux_conv_orth cfgsurp_conv_orth bigramlex_conv_orth trigramlex_conv_orth bigrampos_conv_orth trigrampos_conv_orth
	

Raw data were recorded under two protocols, which are detailed in the publication
 A single EPI sequence, s17-s37
 A multi-echo sequence, s39-s54

To replicate the published analysis:
	1. z-transform ROI timecourses, sndpwr, rate
	2. divide each movement timeseries by its standard deviation
	3. apply exclusion criteria for <subject ROI> pairs detailed in publication supplementary material
	4. construct mixed model (lme4 in R) fitting linguistic and movement parameters against ROI timeseries following model description in publication


Comments? Questions?
jobrenn@umich.edu

Last Updated: 6/9/16

	