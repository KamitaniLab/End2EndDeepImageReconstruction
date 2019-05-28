'''End-to-end deep reconstruction: Data preparation scripts.

Originally developed by Guohua Shen.

This script creates LMDB data of fMRI and images for training of end-to-end deep reconstruction.
'''


import csv
import fnmatch
import glob
import os
from datetime import datetime

import PIL.Image
import caffe
import lmdb
import numpy as np
from scipy.misc import imresize

import bdpy


# Settings ---------------------------------------------------------------

# Image size
img_size = (248, 248)
# For image jittering, we prepare the images to be larger than 227 x 227

# fMRI data
fmri_data_table = [
    {'subject': 'sub-01',
     'data_file': './data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-01'},
    {'subject': 'sub-02',
     'data_file': './data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-02'},
    {'subject': 'sub-03',
     'data_file': './data/fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1',
     'output_dir': './lmdb/sub-03'}
]

# Image data
image_dir = './data/images/training'
image_file_pattern = '*.JPEG'


# Create LMDB data -------------------------------------------------------

for sbj in fmri_data_table:

    print('----------------------------------------')
    print('Subject: %s' % sbj['subject'])
    print('')

    if os.path.exists(sbj['output_dir']):
        print('%s already exists. Skipped.' % sbj['output_dir'])
        continue
    else:
        os.makedirs(sbj['output_dir'])

    # Create LMDB for fMRI data

    # Load fMRI data
    print('Loading %s' % sbj['data_file'])
    fmri_data_bd = bdpy.BData(sbj['data_file'])

    # Load image files
    images_list = glob.glob(os.path.join(image_dir, image_file_pattern))  # List of image files (full path)
    images_table = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in images_list}                                 # Image label to file path table
    label_table = {os.path.splitext(os.path.basename(f))[0]: i + 1
                   for i, f in enumerate(images_list)}                    # Image label to serial number table

    # Get image labels in the fMRI data
    #import pdb; pdb.set_trace()
    fmri_labels = fmri_data_bd.get('Label')[:, 1].flatten()

    # Convet image labels in fMTI data from float to file name labes (str)
    fmri_labels = ['n%08d_%d' % (int(('%f' % a).split('.')[0]),
                                 int(('%f' % a).split('.')[1]))
                   for a in fmri_labels]

    # Get sample indexes
    n_sample = fmri_data_bd.dataset.shape[0]

    index_start = 1
    index_end = n_sample
    index_step = 1

    sample_index_list = range(index_start, index_end + 1, index_step)

    # Shuffle the sample indexes
    sample_index_list = np.random.permutation(sample_index_list)

    # Save the sample indexes
    save_name = 'sample_index_list.txt'
    np.savetxt(os.path.join(sbj['output_dir'], save_name), sample_index_list, fmt='%d')

    # Get fMRI data in the ROI
    fmri_data = fmri_data_bd.select(sbj['roi_selector'])

    # Normalize fMRI data
    fmri_data_mean = np.mean(fmri_data, axis=0)
    fmri_data_std = np.std(fmri_data, axis=0)

    fmri_data = (fmri_data - fmri_data_mean) / fmri_data_std

    map_size = 100 * 1024 * len(sample_index_list) * 10
    env = lmdb.open(os.path.join(sbj['output_dir'], 'fmri'), map_size=map_size)

    with env.begin(write=True) as txn:
        for j0, sample_index in np.ndenumerate(sample_index_list):

            sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
            sample_label_num = label_table[sample_label]  # Sample label (serial number)

            print('Index %d, sample %d' % (j0[0] + 1, sample_index))
            print('Data label: %d (%s)' % (sample_label_num, sample_label))
            print(' ')

            # fMRI data in the sample
            sample_data = fmri_data[sample_index - 1, :]
            sample_data = np.float64(sample_data)  # Datum should be double float (?)
            sample_data = np.reshape(sample_data, (sample_data.size, 1, 1))  # Num voxel x 1 x 1

            datum = caffe.io.array_to_datum(sample_data)
            datum.label = sample_label_num  # Datum.label should be int (int32)
            # The encode is only essential in Python 3
            str_id = '%05d' % (j0[0] + 1)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

    # Create lmdb for images
    print('----------------------------------------')
    print('Images')

    env = lmdb.open(os.path.join(sbj['output_dir'], 'images'), map_size=map_size)

    map_size = 30 * 1024 * len(sample_index_list) * 10

    with env.begin(write=True) as txn:
        for j0, sample_index in np.ndenumerate(sample_index_list):

            sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
            sample_label_num = label_table[sample_label]  # Sample label (serial number)

            print('Index %d, sample %d' % (j0[0] + 1, sample_index))
            print('Data label: %d (%s)' % (sample_label_num, sample_label))
            print(' ')

            # Load images
            image_file = images_table[sample_label]
            img = PIL.Image.open(image_file)
            img = imresize(img, img_size, interp='bilinear')

            # Monochrome --> RGB
            if img.ndim == 2:
                img_rgb = np.zeros((img_size[0], img_size[1], 3))
                img_rgb[:, :, 0] = img
                img_rgb[:, :, 1] = img
                img_rgb[:, :, 2] = img
                img = img_rgb

            # h x w x c --> c x h x w
            img = img.transpose(2, 0, 1)

            # RGB --> BGR
            img = img[::-1]

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = img.shape[0]
            datum.height = img.shape[1]
            datum.width = img.shape[2]
            datum.data = img.tobytes()
            datum.label = sample_label_num

            str_id = '%05d' % (j0[0] + 1)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

print('Done!')
