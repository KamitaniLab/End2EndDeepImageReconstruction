# End-to-End Deep Image Reconstruction

Data, pre-trained models and code for [Shen, Dwivedi, Majima, Horikawa, and Kamitani (2019) End-to-end deep image reconstruction from human brain activity. Front. Comput. Neurosci](https://www.frontiersin.org/articles/10.3389/fncom.2019.00021/full). ([bioRxiv preprint](https://www.biorxiv.org/content/10.1101/272518v1)).

## Requirements

- Python 2.7
- Numpy 
- Scipy
- Pillow (PIL)
- Caffe with up-convolutional layer
    - https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim)

## Usage

### Reconstruct image using pre-trained model

We have released pre-trained models (generator networks of the end-to-end deep image reconstruction models) of the 3 human subjects in our study (see [trained_model/README.md](trained_model/README.md)).

You can use these pre-trained models to reconstruct images from fMRI data (see [fmri_data/README.md](fmri_data/README.md)).

The images are reconstructed by inputting the test fMRI data to the trained generator, and forward passing through the generator net.
We provide example scripts to reconstruct images from human brain fMRI activity (please find the scripts in [model_test/](model_test/)):

- `reconstruct_img_from_fmri_subject_01.py`
- `reconstruct_img_from_fmri_subject_02.py`
- `reconstruct_img_from_fmri_subject_03.py`

### Train your own models

You can train your own models by using the training scripts (see [model_training/README.md](model_training/README.md)).
To do so, you need:

- preprocessed fMRI data and stimulus images (see [fmri_data/README.md](fmri_data/README.md))
- create LMDB data (see [lmdb_data/README.md](lmdb_data/README.md)) 
- pre-trained CNN model (see [bvlc_reference_caffenet/README.md](bvlc_reference_caffenet/README.md))
- set parameters in the training scripts (see [model_training/README.md](model_training/README.md)) 
- run the training script:

      $ cd model_training
      $ python train.py`

## Reference

[1] We used the framework proposed in this article: [Dosovitskiy & Brox (2016) Generating Images with Perceptual Similarity Metrics based on Deep Networks. Advances in Neural Information Processing Systems (NIPS).](http://lmb.informatik.uni-freiburg.de//Publications/2016/DB16c)

The article is available at: http://arxiv.org/abs/1602.02644

[2] The code for training models are modified from the released code of the above article, and the original code are available at: https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.5.zip

[3] The code for creating LMDB data are based on the example code shared in: http://deepdish.io/2015/04/28/creating-lmdb-in-python/

[4] For the details of fMRI preprocessing, please refer to: Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity, http://dx.doi.org/10.1371/journal.pcbi.1006633

## Author

Shen Guo-Hua (E-mail: shen-gh@atr.jp)

## Acknowledgement

The author thanks precious discussion and advice from the members in DNI (http://www.cns.atr.jp/dni/) and Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/).
The author thanks Mitsuaki Tsukamoto for software installation and computational environment setting.
The author thanks Tomoyasu Horikawa for fMRI data preprocessing.
The author thanks Shuntaro Aoki for data curation and example code to read fMRI data.
