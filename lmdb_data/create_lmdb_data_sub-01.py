# create lmdb data for fmri and image
# create 2 lmdb files for fmri and image data
# the order of the samples are the same for the 2 lmdb data files
# by Shen Guo-Hua (shen-gh@atr.jp), 2019-03-26


# import
import os
import lmdb
import caffe
import bdpy
import fnmatch
import PIL.Image
import numpy as np
from scipy.misc import imresize
from datetime import datetime


# parameters and setting-----------------------------------------------------------------------------
img_size = (248,248)  # img size
# for the purpose of image jittering, we prepare the images to be larger than 227 x 227

img_dir = 'path/to/directory/of/training/images'  # the directory for all the traning images
img_file_suffix = '.JPEG'  # the suffix of image file

fmri_data_file = '../fmri_data/sub-01_perceptionNaturalImageTraining_original_VC.h5'  # the file of the fmri data
ROI_selector_str = 'ROI_VC = 1'  # Data select expression specifying columns (ROIs) used as the training fmri data
label_str = 'stimulus_id'  # specifying columns used as the label of the training fmri data

save_dir = '.'  # the directory for saving the resulting lmdb data
#----------------------------------------------------------------------------------------------------


# get the image file list (full file name: full path + file name) and the image file name list (only the file name of the image)
img_file_list = []
img_file_name_list = []
for root, _, fn_list in os.walk(img_dir):
    for fn in fnmatch.filter(fn_list, '*'+img_file_suffix):
        img_file_list.append(os.path.join(root, fn))
        img_file_name_list.append(fn)


# get fMRI data and labels (the corresponding image names)
print('Loading %s' % fmri_data_file)
fmri_data_bd = bdpy.BData(fmri_data_file)
fmri_data = fmri_data_bd.select(ROI_selector_str)
fmri_labels = fmri_data_bd.get(label_str).flatten()
num_of_sample = fmri_data_bd.dataset.shape[0]


# Normalize fMRI data
fmri_data_mean = np.mean(fmri_data, axis=0)
fmri_data_std = np.std(fmri_data, axis=0)
fmri_data = (fmri_data - fmri_data_mean) / fmri_data_std


# Convet fMTI labels from float to file name labes (str)
fmri_labels = ['n%08d_%d' % (int(('%f' % label).split('.')[0]), int(('%f' % label).split('.')[1])) + img_file_suffix for label in fmri_labels]


# sample index list
index_start = 1
index_end = num_of_sample
index_step = 1
sample_index_list = range(index_start, index_end+1, index_step)


# index permutation (to shuffle the order of the samples)
sample_index_list = np.array(sample_index_list)
sample_index_list = np.random.permutation(sample_index_list)


# make sub-directory for saving the resulting lmdb data ('create_lmdb_data_<timestamp>')
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)


# create lmdb for fmri
map_size = 100*1024 * num_of_sample * 10
env = lmdb.open(os.path.join(save_path,'fmri'),map_size=map_size)
with env.begin(write=True) as txn:
    for j0 in xrange(num_of_sample):
        #
        sample_index = sample_index_list[j0]
        print('fmri')
        print(j0+1)
        print(sample_index)
        print(' ')
        
        # fmri data
        fmri_data_j0 = fmri_data[sample_index-1,:]
        
        # convert fmri data format
        fmri_data_j0 = np.float64(fmri_data_j0)
        fmri_data_j0 = np.reshape(fmri_data_j0,(fmri_data_j0.size,1,1))
        
        #
        datum = caffe.io.array_to_datum(fmri_data_j0)
        # The encode is only essential in Python 3
        str_id = '%05d'%(j0+1)
        txn.put(str_id.encode('ascii'),datum.SerializeToString()) 


# create lmdb for image
map_size = 30*1024 * num_of_sample * 10
env = lmdb.open(os.path.join(save_path,'img'),map_size=map_size)
with env.begin(write=True) as txn:
    for j0 in xrange(num_of_sample):
        #
        sample_index = sample_index_list[j0]
        print('img')
        print(j0+1)
        print(sample_index)
        print(' ')
        
        # load img
        fmri_label = fmri_labels[sample_index-1]
        img_index = img_file_name_list.index(fmri_label)
        img_file = img_file_list[img_index]
        img = PIL.Image.open(img_file)
        img = np.asarray(img)
        
        # resize image
        img = imresize(img,img_size,interp='bilinear')
        
        # Monochrome --> RGB
        if len(img.shape) == 2:
            img3 = np.zeros((img_size[0],img_size[1],3), dtype=img.dtype)
            img3[:, :, 0] = img 
            img3[:, :, 1] = img 
            img3[:, :, 2] = img 
            img = img3
        
        # change dimesion from [height, width, color] to [color, height, width]
        img = img.transpose(2,0,1)
        
        # RGB --> BGR
        img = img[::-1]
        
        #
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = img.shape[0]
        datum.height = img.shape[1]
        datum.width = img.shape[2]
        datum.data = img.tobytes() # or .tostring() if numpy < 1.9
        # The encode is only essential in Python 3
        str_id = '%05d'%(j0+1)
        txn.put(str_id.encode('ascii'),datum.SerializeToString())       


#
print('done!')

