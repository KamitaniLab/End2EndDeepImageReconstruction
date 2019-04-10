# Trained models

The pre-trained models (generator networks of the end-to-end deep image reconstruction model) can be downloaded from: https://figshare.com/articles/End-to-end_deep_image_reconstruction_from_human_brain_activity/7916144

These models were trained to reconstruct the stimulus images from human brain fMRI activity, which were trained for 500,000 mini-batch iterations, with batch size of 64.

The pre-trained models of 3 human subjects are released:

- trainedmodel_sub-01

- trainedmodel_sub-02

- trainedmodel_sub-03


In the directory of each subject, there are 2 files:

- the caffemodel file for the trained generator: generator.caffemodel

- the prototxt file for the generator: generator_test.prototxt

