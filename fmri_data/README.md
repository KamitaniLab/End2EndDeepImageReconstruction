## fMRI data

The raw fMRI data is available at: https://openneuro.org/datasets/ds001506

The preprocessed fMRI data is available at: https://figshare.com/articles/Deep_Image_Reconstruction/7033577

The preprocessed fMRI data was obtained by the methodology in our previous study:

Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity, http://dx.doi.org/10.1371/journal.pcbi.1006633


To run the example codes, you need to download the following preprocessed fMRI data:

- sub-01_perceptionNaturalImageTraining_original_VC.h5

- sub-01_perceptionNaturalImageTest_original_VC.h5

- sub-02_perceptionNaturalImageTraining_original_VC.h5

- sub-02_perceptionNaturalImageTest_original_VC.h5

- sub-03_perceptionNaturalImageTraining_original_VC.h5

- sub-03_perceptionNaturalImageTest_original_VC.h5


You can use the script, download_fmri.sh, to download these required files.



## Stimulus images

Each fMRI data sample is labeled by name of the stimulus image.

The name of the stimulus image is converted to a floating number (‘stimulus_id’) as follows:

The integer part of the floating number indicates WordNet ID for the synset (category);

The decimal part of the floating number indicates image ID;

For example, '1518878.005958' represents 'image 5958 in synset n01518878' (‘ostrich’).

We do not include stimulus image files in the open dataset because of license issues.

Please contact us via email (brainliner-admin@atr.jp) for sharing the image files.