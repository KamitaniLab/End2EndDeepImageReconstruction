# Pretrained models

The pre-trained models (generator networks of the end-to-end deep image reconstruction model) can be downloaded from: <https://figshare.com/articles/End-to-end_deep_image_reconstruction_from_human_brain_activity/7916144>.

These models were trained to reconstruct the stimulus images from human brain fMRI activity, which were trained for 500,000 mini-batch iterations, with batch size of 64.

The pre-trained models of 3 human subjects are released:

- [trainedmodel_sub-01.zip](https://ndownloader.figshare.com/files/14985422)
- [trainedmodel_sub-02.zip](https://ndownloader.figshare.com/files/14985485)
- [trainedmodel_sub-03.zip](https://ndownloader.figshare.com/files/14985506)

In the directory of each subject, there are 4 files:

- **generator.caffemodel**: the Caffe caffemodel file for the trained generator
- **generator_test.prototxt**: the Caffe prototxt file for the generator
- **discriminator.caffemodel**: the Caffe caffemodel file for the trained discriminator
- **discriminator.prototxt**: the Caffe prototxt file for the discriminator

To run `end2end_test.py` with the pretrained models, put the files as below:

    net_pretrained
    ├── sub-01
    │   ├── generator.caffemodel
    │   └── generator_test.prototxt
    ├── sub-02
    │   ├── generator.caffemodel
    │   └── generator_test.prototxt
    └── sub-03
        ├── generator.caffemodel
        └── generator_test.prototxt
