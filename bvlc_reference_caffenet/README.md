This is the CaffeNet, which will be used as the comparartor (encoder) net in the end-to-end deep image reconstrcution model.

The parameters of the comparator were fixed throughout the training because it was only used for the comparison in feature space.

The CaffeNet is available at: https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet

The script, downloadnet.sh, will download the required files for the example code:

- the protocol file: deploy.prototxt (https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt)
- the caffemodel file: bvlc_reference_caffenet.caffemodel (http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel)

The protocol file is protocol buffer definition file which defines the architecture of the CNN model.
The caffemodel is a binary protobuf file which contains the pre-trained weights for each layer of the network.