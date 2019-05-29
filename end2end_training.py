'''End-to-end deep reconstruction: Training script

The original version was developed by Guohua Shen.
'''


import os
import shutil
import socket
import time
from argparse import ArgumentParser

import numpy as np
import lmdb

import caffe
from caffe.proto import caffe_pb2

from bdpy.distcomp import DistComp


# Settings ###################################################################

# Data settings
data_table = [
    {'subject':    'sub-01',
     'lmdb_fmri':  './lmdb/sub-01/fmri',
     'lmdb_images': './lmdb/sub-01/images',
     'output_dir': './net_trained/sub-01'},
    {'subject':    'sub-02',
     'lmdb_fmri':  './lmdb/sub-02/fmri',
     'lmdb_images': './lmdb/sub-02/images',
     'output_dir': './net_trained/sub-02'},
    {'subject':    'sub-03',
     'lmdb_fmri':  './lmdb/sub-03/fmri',
     'lmdb_images': './lmdb/sub-03/images',
     'output_dir': './net_trained/sub-03'},
    ]

# Network settings
templates_dir = './net_templates'

encoder_net = './net/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# AlexNet
# Source: http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

# Training settings
max_iter = int(1e6+1) # Maximum number of iterations
batch_size = 64
recogn_feat_shape = (256, 6, 6)

snapshot_every = 10000

# GPU usage (set -1 to use CPU)
gpu_id = 0

# Information display
display_every = 50

# Misc
start_snapshot = 0
tmp_dir = './tmp'

# Main #######################################################################

# Initial setups -------------------------------------------------------------

# GPU setup
if gpu_id >= 0:
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

# Parsing arguments
argparse = ArgumentParser(description=__doc__)
argparse.add_argument('--resume', nargs=1, type=int, dest='resume_snapshot',
                      help='Snapshot from which the training resumes', default=[0])
args = argparse.parse_args()

if args.resume_snapshot[0]:
    start_snapshot = args.resume_snapshot[0]

# Training loop --------------------------------------------------------------

for sbj in data_table:
    print('----------------------------------------')
    print('Subject %s' % sbj['subject'])

    # Data settings
    lmdb_fmri = sbj['lmdb_fmri']
    lmdb_images = sbj['lmdb_images']

    # Output directory
    output_dir = sbj['output_dir']

    net_dir = os.path.join(output_dir, 'net')
    solver_dir = os.path.join(output_dir, 'solver')
    snapshots_dir = os.path.join(output_dir, 'snapshots')

    # Directories setup
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    if not os.path.exists(solver_dir):
        os.makedirs(solver_dir)

    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    comp_id = os.path.splitext(os.path.basename(__file__))[0] + '-' + sbj['subject']
    dist = DistComp(lockdir=tmp_dir, comp_id=comp_id)
    if dist.islocked():
        print('%s is already running. Skipped.' % comp_id)
        continue

    dist.lock()
        
    # Get data parameters ----------------------------------------------------

    # Get the number of voxels
    # FIXME: there must be better ways...
    lmdb_env = lmdb.open(lmdb_fmri)
    with lmdb_env.begin() as txn:
        lmdb_cursor = txn.cursor()
        datum = caffe_pb2.Datum()

        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
            n_voxels = data.shape[0]
            break

    # Network preparation ----------------------------------------------------

    # Create data_fmri.prototxt
    with open(os.path.join(templates_dir, 'data_fmri.prototxt'), 'r') as f:
        data_template = f.read()

    with open(os.path.join(net_dir, 'data_fmri.prototxt'), 'w') as f:
        f.write(data_template.replace('@DATAPATH@', lmdb_fmri))

    # Create data_images.prototxt
    with open(os.path.join(templates_dir, 'data_images.prototxt'), 'r') as f:
        data_template = f.read()

    with open(os.path.join(net_dir, 'data_images.prototxt'), 'w') as f:
        f.write(data_template.replace('@DATAPATH@', lmdb_images))

    # Create generator.prototxt
    with open(os.path.join(templates_dir, 'generator.prototxt'), 'r') as f:
        gen_template = f.read()

    with open(os.path.join(net_dir, 'generator.prototxt'), 'w') as f:
        f.write(gen_template.replace('@NUMVOXELS@', str(n_voxels)))

    # Create generator_test.prototxt
    with open(os.path.join(templates_dir, 'generator_test.prototxt'), 'r') as f:
        gen_template = f.read()

    with open(os.path.join(net_dir, 'generator_test.prototxt'), 'w') as f:
        f.write(gen_template.replace('@NUMVOXELS@', str(n_voxels)))

    # Create encoder.prototxt, and discriminator.prototxt
    shutil.copy(os.path.join(templates_dir, 'encoder.prototxt'), os.path.join(net_dir, 'encoder.prototxt'))
    shutil.copy(os.path.join(templates_dir, 'discriminator.prototxt'), os.path.join(net_dir, 'discriminator.prototxt'))

    # Create solvers
    solvers = ('encoder', 'generator', 'discriminator', 'data_fmri', 'data_images')

    with open(os.path.join(templates_dir, 'solver_template.prototxt'), 'r') as f:
        solver_template = f.read()

    for curr_net in solvers:
        with open(os.path.join(solver_dir, 'solver_%s.prototxt' % curr_net), 'w') as f:
            f.write(solver_template.replace('@NET@', os.path.join(net_dir, curr_net)))

    # Initialize networks
    encoder = caffe.AdamSolver(os.path.join(solver_dir, 'solver_encoder.prototxt'))
    generator = caffe.AdamSolver(os.path.join(solver_dir, 'solver_generator.prototxt'))
    discriminator = caffe.AdamSolver(os.path.join(solver_dir, 'solver_discriminator.prototxt'))
    data_fmri_reader = caffe.AdamSolver(os.path.join(solver_dir, 'solver_data_fmri.prototxt'))
    data_img_reader = caffe.AdamSolver(os.path.join(solver_dir, 'solver_data_images.prototxt'))

    encoder.net.copy_from(encoder_net)

    # Load networks from snapshot (resuming)
    if start_snapshot:
        curr_snapshot_folder = os.path.join(snapshots_dir, str(start_snapshot))
        print('\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n')

        generator_caffemodel = os.path.join(curr_snapshot_folder, 'generator.caffemodel')
        if os.path.isfile(generator_caffemodel):
            generator.net.copy_from(generator_caffemodel)
        else:
            raise Exception('File %s does not exist' % generator_caffemodel)

        discriminator_caffemodel = os.path.join(curr_snapshot_folder, 'discriminator.caffemodel')
        if os.path.isfile(discriminator_caffemodel):
            discriminator.net.copy_from(discriminator_caffemodel)
        else:
            raise Exception('File %s does not exist' % discriminator_caffemodel)

    # Read weights of losses
    recon_loss_weight = generator.net._blob_loss_weights[generator.net._blob_names_index['recon_loss']]
    feat_loss_weight = encoder.net._blob_loss_weights[encoder.net._blob_names_index['feat_recon_loss']]
    discr_loss_weight = discriminator.net._blob_loss_weights[discriminator.net._blob_names_index['discr_loss']]

    # Training ---------------------------------------------------------------

    host_name = socket.gethostname()
    code_name = os.path.realpath(__file__)

    train_discr = True
    train_gen = True

    # Training main loop
    start = time.time()
    for it in range(start_snapshot, max_iter):

        # Read fMRI data
        data_fmri_reader.net.forward_simple()

        # Read image data
        data_img_reader.net.forward_simple()

        # Feed the data (images) to the encoder and run it
        encoder.net.blobs['data'].data[...] = data_img_reader.net.blobs['data'].data
        encoder.net.blobs['feat_gt'].data[...] = np.zeros((batch_size,) + recogn_feat_shape, dtype='float32')
        encoder.net.forward_simple()
        recogn_feat_real = np.copy(encoder.net.blobs['pool5'].data)

        # Feed the data to the generator and run it
        generator.net.blobs['data'].data[...] = data_img_reader.net.blobs['data'].data
        generator.net.blobs['feat'].data[...] = data_fmri_reader.net.blobs['data'].data[:,:,0,0]
        generator.net.forward_simple()
        generated_img = generator.net.blobs['generated'].data
        recon_loss = generator.net.blobs['recon_loss'].data

        # Encode the generated image to compare its features to the features of the input image
        encoder.net.blobs['data'].data[...] = generated_img
        encoder.net.blobs['feat_gt'].data[...] = recogn_feat_real
        encoder.net.forward_simple()
        feat_recon_loss = encoder.net.blobs['feat_recon_loss'].data

        # Run the discriminator on real data
        discriminator.net.blobs['data'].data[...] = data_img_reader.net.blobs['data'].data
        discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
        discriminator.net.forward_simple()
        discr_real_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
        if train_discr:
            discriminator.increment_iter()
            discriminator.net.clear_param_diffs()
            discriminator.net.backward_simple()

        # Run the discriminator on generated data
        discriminator.net.blobs['data'].data[...] = generated_img
        discriminator.net.blobs['label'].data[...] = np.ones((batch_size,1,1,1), dtype='float32')
        discriminator.net.forward_simple()
        discr_fake_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
        if train_discr:
            discriminator.net.backward_simple()
            discriminator.apply_update()

        # Run the discriminator on generated data with opposite labels, to get the gradient for the generator
        discriminator.net.blobs['data'].data[...] = generated_img
        discriminator.net.blobs['label'].data[...] = np.zeros((batch_size,1,1,1), dtype='float32')
        discriminator.net.forward_simple()
        discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['discr_loss'].data)
        if train_gen:
            generator.increment_iter()
            generator.net.clear_param_diffs()
            encoder.net.backward_simple()
            discriminator.net.backward_simple()
            generator.net.blobs['generated'].diff[...] = encoder.net.blobs['data'].diff + discriminator.net.blobs['data'].diff
            generator.net.backward_simple()
            generator.apply_update()

        # Display info
        if it % display_every == 0:
            print('========================================')
            print('Host name:   ' + host_name)
            print('GPU ID:      ' + str(gpu_id))
            print('Script name: ' + code_name)
            print('')
            print('[%s] Iteration %d: %f seconds' % (time.strftime('%c'), it, time.time()-start))
            print('  recon loss: %e * %e = %f' % (recon_loss, recon_loss_weight, recon_loss*recon_loss_weight))
            print('  feat loss: %e * %e = %f' % (feat_recon_loss, feat_loss_weight, feat_recon_loss*feat_loss_weight))
            print('  discr real loss: %e * %e = %f' % (discr_real_loss, discr_loss_weight, discr_real_loss*discr_loss_weight))
            print('  discr fake loss: %e * %e = %f' % (discr_fake_loss, discr_loss_weight, discr_fake_loss*discr_loss_weight))
            print('  discr fake loss for generator: %e * %e = %f' % (discr_fake_for_generator_loss, discr_loss_weight, discr_fake_for_generator_loss*discr_loss_weight))
            start = time.time()

        # Save snapshot
        if it % snapshot_every == 0:
            curr_snapshots_dir = snapshots_dir +'/' + str(it)
            print('\n === Saving snapshot to ' + curr_snapshots_dir + ' ===\n')
            if not os.path.exists(curr_snapshots_dir):
                os.makedirs(curr_snapshots_dir)
            generator_caffemodel = curr_snapshots_dir + '/' + 'generator.caffemodel'
            generator.net.save(generator_caffemodel)
            discriminator_caffemodel = curr_snapshots_dir + '/' + 'discriminator.caffemodel'
            discriminator.net.save(discriminator_caffemodel)

        # Switch optimizing discriminator and generator, so that neither of them overfits too much
        discr_loss_ratio = (discr_real_loss + discr_fake_loss) / discr_fake_for_generator_loss
        if discr_loss_ratio < 1e-1 and train_discr:
          train_discr = False
          train_gen = True
          print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
        if discr_loss_ratio > 5e-1 and not train_discr:
          train_discr = True
          train_gen = True
          print(' <<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))
        if discr_loss_ratio > 1e1 and train_gen:
          train_gen = False
          train_discr = True
          print('<<< real_loss=%e, fake_loss=%e, fake_loss_for_generator=%e, train_discr=%d, train_gen=%d >>>' % (discr_real_loss, discr_fake_loss, discr_fake_for_generator_loss, train_discr, train_gen))

    dist.unlock()
          
print('Done!')
