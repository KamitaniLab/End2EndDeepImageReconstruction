# reconstruct image by inputing the test fmri to the trained generator;
# the test fmri is normalized using the mean and std of training fMRI data;
# the test fmri re-scaled by sqrt(n), where n is the number of trials agveraged for test fmri data;

# by Shen Guo-Hua (shen-gh@atr.jp), 2019-03-27


# import
import os
import numpy as np
import caffe
import bdpy
import scipy.io as sio
import PIL.Image
from datetime import datetime


# parameters and settings ----------------------------------------------------------------------------------------------------
generator_caffemodel_file = '../trained_model/trainedmodel_sub-01/generator.caffemodel'  # the caffemodel file for the trained generator
generator_prototxt_file = '../trained_model/trainedmodel_sub-01/generator_test.prototxt'  # the prototxt file for the generator
input_layer = 'feat'  # the name of the input layer of the generator
output_layer = 'deconv0'  # the name of the output layer of the generator

test_fmri_data_file = '../fmri_data/sub-01_perceptionNaturalImageTest_original_VC.h5'  # the file of the test fmri data
training_fmri_data_file = '../fmri_data/sub-01_perceptionNaturalImageTraining_original_VC.h5'  # the file of the training fmri data
ROI_selector_str = 'ROI_VC = 1'  # Data select expression specifying columns (ROIs) used as the test and training data
scale = 4.9 # sqrt(n), where n is the number of trials agveraged for test fmri data: 24

img_size = (227,227,3)  # the designed size of the reconstructed image during model training (the central region of the output of the generator [256,256,3])

img_mean = np.float32([104., 117., 123.])  # the image mean for each of the BGR color channels, which are subtracted from the images before training

save_dir = '.'  # the directory for saving the resulting lmdb data

# -----------------------------------------------------------------------------------------------------------------------------


# Load the generator
gen = caffe.Net(generator_prototxt_file,generator_caffemodel_file,caffe.TEST)


# Get the fmri data
print('Loading test fMRI data: %s' % test_fmri_data_file)
fmri_data_bd = bdpy.BData(test_fmri_data_file)
test_fmri_data = fmri_data_bd.select(ROI_selector_str)
test_fmri_labels = fmri_data_bd.get('stimulus_id').flatten()

print('Loading training fMRI data: %s' % training_fmri_data_file)
fmri_data_bd = bdpy.BData(training_fmri_data_file)
training_fmri_data = fmri_data_bd.select(ROI_selector_str)


# Average test fMRI data across the trials corresponding to the same stimulus image
test_fmri_data_avg = []
img_file_name_list = []
unique_labels = np.unique(test_fmri_labels)
for label in unique_labels:
    sample_index = test_fmri_labels==label
    test_fmri_data_sample = test_fmri_data[sample_index,:]
    test_fmri_data_avg.append(np.mean(test_fmri_data_sample, axis=0))
    
    img_file_name = 'n%08d_%d' % (int(('%f' % label).split('.')[0]), int(('%f' % label).split('.')[1]))  # Convet fMTI labels from float ('stimulus_id') to file name labes (str)
    img_file_name_list.append(img_file_name)
    
test_fmri_data_avg = np.vstack(test_fmri_data_avg)
num_of_sample = test_fmri_data_avg.shape[0]


# Normalize and re-scale test fMRI data
fmri_data_mean = np.mean(training_fmri_data, axis=0)
fmri_data_std = np.std(training_fmri_data, axis=0)
test_fmri_data_avg = (test_fmri_data_avg - fmri_data_mean) / fmri_data_std
test_fmri_data_avg = test_fmri_data_avg * scale


# mean img
mean_img = np.zeros(img_size,dtype='float32')
mean_img[:,:,0] = img_mean[2].copy()
mean_img[:,:,1] = img_mean[1].copy()
mean_img[:,:,2] = img_mean[0].copy()


# top left offset for cropping the output image to get the 227x227 image
img_size0 = gen.blobs[output_layer].data.shape[2:]  # the original size of the output of the generator
top_left = ((img_size0[0] - img_size[0])/2,(img_size0[1] - img_size[1])/2)


# make sub-directory for saving the reconstructed images
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)


# functions
def img_deprocess(img,mean_img=None):
    '''convert from Caffe's input image layout'''
    if mean_img is None or mean_img==[] or (isinstance(mean_img,np.ndarray) and mean_img.size==0):
        mean_img = np.float32([123.0, 117.0, 104.0])
        mean_img = np.reshape(mean_img,(1,1,3))
        img_size = img.shape
        mean_img = np.tile(mean_img,[img_size[1],img_size[2],1])
    return np.dstack(img[::-1]) + mean_img

def normalise_img(img):
    '''Normalize the image.
    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int
    '''
    img = img - img.min()
    if img.max()>0:
        img = img * (255.0/img.max())
    img = np.uint8(img)
    return img

def clip_extreme_value(img, pct=1):
    '''clip the pixels with extreme values'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img = np.clip(img, np.percentile(img, pct/2.),
                  np.percentile(img, 100-pct/2.))
    return img


# main
for img_index in range(num_of_sample):
    #
    print('img_index='+str(img_index))
    
    # input data
    input_data = test_fmri_data_avg[img_index,:]
    
    # gen img
    gen.blobs[input_layer].data[0] = input_data.copy()
    gen.forward()
    gen_img = gen.blobs[output_layer].data[0].copy()
    gen_img = gen_img[:,top_left[0]:top_left[0]+img_size[0],top_left[1]:top_left[1]+img_size[1]]
    gen_img = img_deprocess(gen_img,mean_img)
    
    # save
    save_name = img_file_name_list[img_index] + '.mat'
    sio.savemat(os.path.join(save_path,save_name),{'gen_img':gen_img})
    
    # To better display the image, clip pixels with extreme values (0.02% of
    # pixels with extreme low values and 0.02% of the pixels with extreme high
    # values). And then normalise the image by mapping the pixel value to be
    # within [0,255]
    save_name = img_file_name_list[img_index] + '.tif'
    PIL.Image.fromarray(normalise_img(clip_extreme_value(gen_img, pct=0.04))).save(os.path.join(save_path,save_name))
    

##
print('done!')
