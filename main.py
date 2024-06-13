import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

import tensorflow as tf
import pdb

from utils import *
from model_sart_unet import ReconNet
import SynoGenerator
import math
import sys

from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D

from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan, primitives_2d, primitives_3d
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d
from pyronn.ct_reconstruction.layers import projection_3d

num_epochs                    = np.int64(sys.argv[1])             
batch_size                    = np.int64(sys.argv[2]) 
lr_start                      = float(sys.argv[3]) 
lr_end                        = float(sys.argv[4]) 
use_gpu                       = np.int64(sys.argv[5]) 
phase                         = str(sys.argv[6]) 
ckpt_dir                      = str(sys.argv[7]) 
sample_dir                    = str(sys.argv[8])  
ndct_test_dir                 = str(sys.argv[9]) 
svct_test_dir                 = str(sys.argv[10]) 
test_dir                      = str(sys.argv[11]) 
ndct_train_dir                = str(sys.argv[12]) 
svct_train_dir                = str(sys.argv[13]) 
dx                            = float(sys.argv[14]) 
numpix                        = [np.int64(sys.argv[15]),np.int64(sys.argv[15])]
volume_spacing                = [float(sys.argv[16]), float(sys.argv[16])] 
angular_range                 = np.int64(sys.argv[17]) 
numbin                        = np.int64(sys.argv[18]) 
fan_angle                     = float(sys.argv[19]) 
detector_spacing              = float(sys.argv[20]) 
numtheta                      = np.int64(sys.argv[21]) 
geom_type                     = str(sys.argv[22]) 
source_detector_distance      = np.int64(sys.argv[23]) 
source_isocenter_distance     = np.int64(sys.argv[24]) 
strategy_type                 = str(sys.argv[25]) 
numits                        = np.int64(sys.argv[26]) 
ns                            = np.int64(sys.argv[27])
"""
Chooses geometry based on information
Precondition:
- geom_type: string that specified the geometric type (parallel, fanflat, or cone)
- volume_size: size of volume
- volume_shape: shape of volume
- volume_spacing: spacing of volume
- angular_range: angular range
- angular_range_PI: angular range as PI term
- detector_shape: shape of detector
- detector_spacing: spacing of detector
- projection_number: number of projections
- source_detector_distance: detector distance from source
- source_isocenter_distance: detector isocenter distance
Postcondition:
- It returns a geometry type based on information above
"""

def chooseGeometry(geom_type, volume_size, volume_spacing, angular_range_PI, numbin,detector_spacing, projection_number, source_detector_distance, source_isocenter_distance):
    if geom_type == 'parallel':
        geo = GeometryParallel2D(volume_size,volume_spacing,numbin, detector_spacing,projection_number, angular_range_PI)
        geo.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
        return geo
    elif geom_type == 'fanflat':
        geo = GeometryFan2D(volume_size,volume_spacing,numbin,detector_spacing,projection_number, angular_range_PI, source_detector_distance,source_isocenter_distance)
        geo.set_central_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
        return geo
    else:
        geo = GeometryCone3D(volume_size,volume_spacing,numbin, detector_spacing, projection_number, angular_range_PI, source_detector_distance,source_isocenter_distance)
        geo.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geo)) 
        return geo




'''
Tests ReconNet
Precondition: network to load information and test into
Postcondition: No return (reconNet passed is trainned)
'''
def ReconNet_test(ReconNet):
    svct_files= sorted(glob( svct_test_set))
    svct_files= load_floats(svct_files)
    ndct_files= sorted(glob( ndct_test_set))
    ndct_files= load_floats(ndct_files)
    ReconNet.test(svct_files, ndct_files, ckpt_dir= ckpt_dir, save_dir= test_dir)

'''
Load in sinograms
Precondition: 
- svct_train_dir: a string with the path to get data for svct_train related data
- svct_test_dir: a string with the path to get data for svct_test related data
- ndct_train_dir: a string with the path to get data for ndct_train related data
- ndct_test_dir: a string with the path to get data for ndct_test related data
Postcondition: in the following order,
- svct_train_data: array with svct training data
- svct_test_data: array with svct testing data
- ndct_train_data: array with ndct training data
- ndct_test_data: array with ndct testing data
'''
def load_data(svct_train_dir, svct_test_dir, ndct_train_dir, ndct_test_dir):
    svct_train_data = []
    svct_test_data = []
    ndct_train_data = []
    ndct_test_data = []
    
    if (os.path.isdir(svct_train_dir) & os.path.isdir(svct_test_dir) & os.path.isdir(ndct_train_dir) & os.path.isdir(ndct_test_dir)) :		#generate list of filenames from directory
        svct_train_fnames = sorted(glob(svct_train_dir + '/*.flt'))
        svct_test_fnames = sorted(glob(svct_test_dir + '/*.flt'))
        ndct_train_fnames = sorted(glob(ndct_train_dir + '/*.flt'))
        ndct_test_fnames = sorted(glob(ndct_test_dir + '/*.flt'))
        
        print("Loading svct_train_data...\n")
        for files in svct_train_fnames:
            svct_train_float = np.fromfile(files,dtype='f')
            svct_train_data.append(svct_train_float.reshape(numtheta,numbin))
        print("Loading svct_test_data...\n")
        for files in svct_test_fnames:
            svct_test_float = np.fromfile(files,dtype='f')
            svct_test_data.append(svct_test_float.reshape(numtheta,numbin))
        print("Loading ndct_train_data...\n")
        for files in ndct_train_fnames:
            ndct_train_float = np.fromfile(files,dtype='f')
            ndct_train_data.append(ndct_train_float.reshape(512,512))
        print("Loading ndct_test_data...\n")
        for files in ndct_test_fnames:
            ndct_test_float = np.fromfile(files,dtype='f')
            ndct_test_data.append(ndct_test_float.reshape(512,512))
    else:							#single filename
        print("Error: Invalid directory for data")
    return np.array(svct_train_data), np.array(svct_test_data), np.array(ndct_train_data), np.array(ndct_test_data)

'''
main executable of the class
'''
def main(_):
    print("Starting Main...\n")
    if  geom_type == 'fanflat':
        ft = np.tan( np.deg2rad( fan_angle / 2) )    #compute tan of 1/2 the fan angle
        detector_spacing = 2 *  source_detector_distance * ft /  numbin  #width of one detector pixel, calculated based on fan angle
    # create Geometry class
    angular_range_PI = np.radians(angular_range)
    geometry = chooseGeometry( geom_type,  numpix, 
                                     volume_spacing,  angular_range_PI, 
                                     numbin,  detector_spacing,  numtheta,
                                     source_detector_distance,  source_isocenter_distance)

    if not os.path.exists( ckpt_dir):
        os.makedirs( ckpt_dir)
    if not os.path.exists( sample_dir):
        os.makedirs( sample_dir)
    if not os.path.exists( test_dir):
        os.makedirs( test_dir)


    # defines a decaying learning rate for each epoch
    lr = []
    start = 1
    end = math.log( lr_end, lr_start)
    step = (end - start)/num_epochs
    for i in np.arange(start, end, step):
        current_lr =  lr_start**i
        lr.append(current_lr)

    #lr = lr_start * np.ones([num_epochs])
    #lr[30:] = lr[0] / 10.0

    svct_train_data, svct_test_data, ndct_train_data, ndct_test_data = load_data(svct_train_dir, svct_test_dir, ndct_train_dir, ndct_test_dir)
    # print(sino)
    if  use_gpu:
        # added to control the gpu memory
        print("Loading onto GPU...\n")
        # print(sino[0], " ", "sino0")
        # print(tf.shape(sino[0]), " ", "sino0 shape")

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            model = ReconNet(sess,
                numpix,dx,numbin,numtheta,numits,
                geometry, num_epochs, strategy_type)
            if  phase == 'train':
                print("Starting Training Routine...")
                model.train(ndct_train_data, svct_train_data, ndct_test_data, svct_test_data,
                            batch_size, ckpt_dir, num_epochs, lr, sample_dir)
            elif  phase == 'test':
                ReconNet_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = ReconNet(sess,
                    numpix,dx,numbin,numtheta,numits,
                    geometry, num_epochs, strategy_type)
            if  phase == 'train':
                model.train(ndct_train_data, svct_train_data, ndct_test_data, svct_test_data, batch_size, ckpt_dir, num_epochs, lr, sample_dir)
            elif  phase == 'test':
                ReconNet_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()

