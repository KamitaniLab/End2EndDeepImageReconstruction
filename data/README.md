# End-to-End Deep Image Reconstruction: fMRI data and stimulus images

## fMRI data

To run the example code, you need to download the following preprocessed fMRI data files:

- sub-01_perceptionNaturalImageTraining_original_VC.h5
- sub-01_perceptionNaturalImageTest_original_VC.h5
- sub-02_perceptionNaturalImageTraining_original_VC.h5
- sub-02_perceptionNaturalImageTest_original_VC.h5
- sub-03_perceptionNaturalImageTraining_original_VC.h5
- sub-03_perceptionNaturalImageTest_original_VC.h5

All fMRI data files should be placed in `data/fmri` directory.

You can download the preprocessed fMRI data at <https://figshare.com/articles/Deep_Image_Reconstruction/7033577>.
Instead, you can use the script, `download_fmri.sh`, to download these required files.

Usage:

``` shellsession
$ bash download_fmri.sh
```

The raw (unpreprocessed) fMRI data are available at <https://openneuro.org/datasets/ds001506>.

The fMRI data were obtained and preprocessed by the methodology in our previous study:

Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. PLOC Comput Biol. <http://dx.doi.org/10.1371/journal.pcbi.1006633>

## Stimulus images

The stimulus images used in [Horikawa & Kamitani, 2017](https://www.nature.com/articles/ncomms15037) were also used in this research.
We do not include stimulus image files in the open dataset because of license issues.
You can request us the images via https://forms.gle/ujvA34948Xg49jdn9.

Training and test stimulus images should be placed in `data/images/trainig` and `data/images/test`, respectively.

Each fMRI data sample is labeled by name of the stimulus image. The name of the stimulus image is converted to a floating number (‘stimulus_id’) as follows:

- The integer part of the floating number indicates WordNet ID for the synset (category);
- The decimal part of the floating number indicates image ID;

For example, '1518878.005958' represents 'image 5958 in synset n01518878' (‘ostrich’).
