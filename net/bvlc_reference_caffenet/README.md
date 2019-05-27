This is the [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet), which will be used as the comparartor (encoder) net in the end-to-end deep image reconstrcution model.

The parameters of the comparator were fixed throughout the training because it was only used for the comparison in feature space.

The following two files are required for the example code.

- The protocol file: `deploy.prototxt` (<https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt>; included in this repository)
- The caffemodel file: `bvlc_reference_caffenet.caffemodel` (<http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel>)

The protocol file is protocol buffer definition file which defines the architecture of the CNN model.
The caffemodel is a binary protobuf file which contains the pre-trained weights for each layer of the network.

The script, `downloadnet.sh`, will download the caffemodel files.

Useage:

``` shellsession
$ bash downloadnet.sh caffenet
```
