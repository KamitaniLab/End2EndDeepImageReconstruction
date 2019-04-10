## Test the trained model

Once the end-to-end deep image reconstruction model was trained, you can test the trained model on test datasets, and directly reconstruct image using an fMRI pattern as the input.

The images are reconstructed by inputing the test fmri data to the trained generator, and forward passing through the generator net.

We provide example codes to reconstruct images from human brain fMRI activity using the trained models we released:

- the example codes for subject 01:  reconstruct_img_from_fmri_sub-01.py

- the example codes for subject 02:  reconstruct_img_from_fmri_sub-02.py

- the example codes for subject 03:  reconstruct_img_from_fmri_sub-03.py


To run these codes, you need:

- pre-trained model (generator) (see trained_model/README.md)

- preprocessed fMRI data (see fmri_data/README.md) 

