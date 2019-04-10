## LMDB data

We save the training fmri data and stimulus images as LMDB format.

LMDB is the database of choice when using Caffe with large datasets (http://deepdish.io/2015/04/28/creating-lmdb-in-python/).

To create LMDB data, please use the codes for each of the 3 human subjects:

- create_lmdb_data_sub-01.py

- create_lmdb_data_sub-02.py

- create_lmdb_data_sub-03.py


To run these codes, you need: the preprocessed fmri data and the stimulus images (see fmri_data/README.md).

The codes will create 2 LMDB files for the training fmri and image data.

The orders of the samples are matched in the 2 lmdb files.
