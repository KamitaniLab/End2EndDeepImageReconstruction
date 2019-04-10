# Training settings and codes 

This repository contains all the settings and codes for training an end-to-end deep image reconstruction model to reconstruct images from human brain fMRI activity.

All the files and codes in this repository are modified from the released codes of the article:
[Dosovitskiy & Brox (2016) Generating Images with Perceptual Similarity Metrics based on Deep Networks. Advances in Neural Information Processing Systems (NIPS).](http://lmb.informatik.uni-freiburg.de//Publications/2016/DB16c)

The article is available at: http://arxiv.org/abs/1602.02644

The original codes are available at: https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.5.zip (training/fc6)


## Modification to the original training codes:

- use fmri data as input of the generator net

- do not feed fmri to the discriminator

- randomly crop 227x227 image from 248x248 image


## The files in this repository

- the Caffe protocol files, which defines the architectures of the networks in the model: 

    - `generator.prototxt` (the generator net)
    
    - `discriminator.prototxt` (the discriminator net)
    
    - `encoder.prototxt` (the encoder or comparator net)
    
    - `fmri_data.prototxt` (only 1 input layer for fmri data)
    
    - `img_data.prototxt` (only 1 input layer for image data)

- the Caffe protocol file, which defines the training configuration: `solver_template.prototxt`

- training codes: `train.py`


## Training new models

- Compile modified Caffe version to train your own models

    - https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim)

- lmdb files for the training fmri and image data (see lmdb_data/README.md)

- parameters and settings

    - set the directory of the corresponding lmdb file for the training fmri and image data in the files: `fmri_data.prototxt` and `img_data.prototxt`
    
    - set the number of voxels in the file: `generator.prototxt`
    
    - set the parameters in the training codes: `train.py`

- run the training codes:

    - `cd model_training`
    
    - `python train.py`
