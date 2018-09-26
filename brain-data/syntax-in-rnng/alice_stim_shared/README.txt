Alice fMRI Stimuli
Jonathan Brennan <jobrenn@umich.edu>
0000-0002-3639-350X

These files contain stimuli used in:

Brennan, J. R., Stabler, E. P., Van Wagenen, S. E., Luh, W.-M., & Hale, J. T. (2016). Abstract linguistic structure correlates with temporal activity during naturalistic comprehension. Brain Lang, 157-158, 81–94.


Audiobook file from

http://librivox.org/alices-adventures-in-wonderland-by-lewis-carroll/

Ch. 1

Data processing

1. converted to mono (audacity)
2. dilate (praat algorithm) 120% to ~12 minutes
3. noise-reduction (audacity algorithm), 16dB & default settings
4. 20 s silence padding added to beginning (audacity)
5. normalized to 70 dB (praat)


TextGrid creation

1. Penn Forced Aligner (http://fave.ling.upenn.edu/FAAValign.html)

with inputs: 
	1. raw audiobook file 
	2. text of Ch. 1 from https://www.gutenberg.org/ebooks/11

2. Dilate text-grid times (step 2 above) and pre-prend 20 s (step 4 above)
3. Alignments adjusted word-by-word by hand

Last Updated: 12/12/13
