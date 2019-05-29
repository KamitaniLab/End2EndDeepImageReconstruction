'''End-to-end deep reconstruction: Test (image reconstruction) script

Originally developed by Guohua Shen.
'''


import os
from datetime import datetime

import PIL.Image
import caffe
import numpy as np
import scipy.io as sio
from scipy.misc import imresize

import bdpy


# Settings ###################################################################

# Output dir
output_dir_base = os.path.join('./results', os.path.splitext(os.path.basename(__file__))[0] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S'))

# Data settings
data_table = [
    {
        'subject': 'sub-01',

        # The file of the test fmri data
        'data_fmri_test': './data/fmri/sub-01_perceptionNaturalImageTest_original_VC.h5',

        # The file of the training fmri data
        'data_fmri_training': './data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',

        # Path to caffemodel trained generator (caffemodel and prototxt files).
        # 'generator_caffemodel' can be either path to a caffemodel file or
        # a directory that contains snapshots of trained models.
        'generator_caffemodel': './net_pretrained/sub-01/generator.caffemodel',
        'generator_prototxt': './net_pretrained/sub-01/generator_test.prototxt',
        # 'generator_caffemodel': './net_trained/sub-01/snapshots',
        # 'generator_prototxt': './net_trained/sub-01/net/generator_test.prototxt',

        # Data select expression specifying columns (ROIs) used as the test and training data
        'roi_selector': 'ROI_VC = 1',

        # Output directory
        'output_dir': os.path.join(output_dir_base, 'sub-01')
    },
    {
        'subject': 'sub-02',
        'data_fmri_test': './data/fmri/sub-02_perceptionNaturalImageTest_original_VC.h5',
        'data_fmri_training': './data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
        'generator_caffemodel': './net_pretrained/sub-02/generator.caffemodel',
        'generator_prototxt': './net_pretrained/sub-02/generator_test.prototxt',
        # 'generator_caffemodel': './net_trained/sub-02/snapshots',
        # 'generator_prototxt': './net_trained/sub-02/net/generator_test.prototxt',
        'roi_selector': 'ROI_VC = 1',
        'output_dir': os.path.join(output_dir_base, 'sub-02')
    },
    {
        'subject': 'sub-03',
        'data_fmri_test': './data/fmri/sub-03_perceptionNaturalImageTest_original_VC.h5',
        'data_fmri_training': './data/fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
        'generator_caffemodel': './net_pretrained/sub-03/generator.caffemodel',
        'generator_prototxt': './net_pretrained/sub-03/generator_test.prototxt',
        # 'generator_caffemodel': './net_trained/sub-03/snapshots',
        # 'generator_prototxt': './net_trained/sub-03/net/generator_test.prototxt',
        'roi_selector': 'ROI_VC = 1',
        'output_dir': os.path.join(output_dir_base, 'sub-03')
    },
]

scale = 4.9 # sqrt(n), where n is number of averaged trials

# Generator settings
input_layer = 'feat'      # The name of the input layer of the generator
output_layer = 'deconv0'  # The name of the output layer of the generator

# Image settings

img_mean = np.float32([104., 117., 123.])  # The image mean for each of the BGR color channels, which are subtracted from the images before training
img_size = (227, 227, 3)  # The designed size of the reconstructed image during model training (the central region of the output of the generator [256,256,3])


# Functions ##################################################################

def img_deprocess(img, mean_img=None):
    if mean_img is None or mean_img == [] or (isinstance(mean_img, np.ndarray) and mean_img.size == 0):
        mean_img = np.float32([123.0, 117.0, 104.0])
        mean_img = np.reshape(mean_img,(1, 1, 3))
        img_size = img.shape
        mean_img = np.tile(mean_img, [img_size[1], img_size[2], 1])
    return np.dstack(img[::-1]) + mean_img

def normalize_image(img):
    img = img - img.min()
    if img.max() > 0:
        img = img * (255.0 / img.max())
    img = np.uint8(img)
    return img


# Main #######################################################################

for dat in data_table:

    fmri_data_test_file = dat['data_fmri_test']
    fmri_data_selector = dat['roi_selector']
    fmri_data_training_file = dat['data_fmri_training']
    output_dir = dat['output_dir']
    
    # Setup directory ------------------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load fMRI data -------------------------------------------------------------

    fmri_data = bdpy.BData(fmri_data_test_file)
    brain_data = fmri_data.select(fmri_data_selector)
    brain_labels = fmri_data.select('Label = 1')[:, 1]

    fmri_data_train = bdpy.BData(fmri_data_training_file)
    brain_data_train = fmri_data_train.select(fmri_data_selector)

    brain_data_mean = np.mean(brain_data_train, axis=0)
    brain_data_norm = np.std(brain_data_train, axis=0)

    del(fmri_data_train)
    del(brain_data_train)

    # Average fMRI data across trials
    brain_data_ave = []
    image_list = []

    labels_set = np.unique(brain_labels)
    for lb in labels_set:
        sample_index = brain_labels == lb
        brain_data_sample = brain_data[sample_index, :]
        brain_data_ave.append(np.mean(brain_data_sample, axis=0))

        # Convert stimulus ID to stimulus file name
        image_name = 'n%08d_%d' % (int(('%f' % lb).split('.')[0]),
                                   int(('%f' % lb).split('.')[1]))
        image_list.append(image_name)

    brain_data_ave = np.vstack(brain_data_ave)

    # Normalize fMRI data
    brain_data_ave = (brain_data_ave - brain_data_mean) / brain_data_norm

    # Image preparation ----------------------------------------------------------

    # Image mean
    mean_img = np.zeros(img_size,dtype='float32')
    mean_img[:,:,0] = img_mean[2].copy()
    mean_img[:,:,1] = img_mean[1].copy()
    mean_img[:,:,2] = img_mean[0].copy()

    # Image generation loop ------------------------------------------------------

    if os.path.isfile(dat['generator_caffemodel']):
        # No snapshots
        is_snapshot = False
        snapshot_list = [None]
    else:
        # Get snapshots
        is_snapshot = True
        snapshot_list = [int(n) for n in os.listdir(dat['generator_caffemodel'])]
        snapshot_index_digits = len(str(snapshot_list[-1]))
        
    for snapshot in snapshot_list:

        # Load generator net
        if is_snapshot:
            generator_model = os.path.join(dat['generator_caffemodel'], str(snapshot), 'generator.caffemodel')
        else:
            generator_model = dat['generator_caffemodel']
        generator_prototxt = dat['generator_prototxt']

        generator = caffe.Net(generator_prototxt, generator_model, caffe.TEST)

        # Get top-left position of the image
        img_size0 = generator.blobs[output_layer].data.shape[2:]  # The original size of the output of the generator
        top_left = ((img_size0[0] - img_size[0]) / 2, (img_size0[1] - img_size[1]) / 2)

        # Images loop
        for i, img in enumerate(image_list):
            print('----------------------------------------')
            print('Snapshot: ' + str(snapshot))
            print('Image:    ' + img)
            print('')

            # Prepare fMRI data
            input_data = brain_data_ave[i, :] * scale

            # Generate an image
            generator.blobs[input_layer].data[0] = input_data.copy()
            generator.forward()
            gen_img = generator.blobs[output_layer].data[0].copy()
            gen_img = gen_img[:,top_left[0]:top_left[0]+img_size[0],top_left[1]:top_left[1]+img_size[1]]
            gen_img = img_deprocess(gen_img,mean_img)

            # Save the generated image
            if is_snapshot:
                output_image_dir = os.path.join(output_dir, 'image-' + img)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                output_file_base = 'snapshot-' + str(snapshot).zfill(snapshot_index_digits)
            else:
                output_image_dir = output_dir
                output_file_base = 'image-' + img

            sio.savemat(os.path.join(output_image_dir, output_file_base + '.mat'), {'gen_img': gen_img})
            PIL.Image.fromarray(normalize_image(gen_img)).save(os.path.join(output_image_dir, output_file_base + '.jpg'))

print('Done!')
