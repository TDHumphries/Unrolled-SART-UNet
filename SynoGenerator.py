#Used for executing analysis:
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb                      #Debugger
#Used for file processing:
import os
from glob import glob
import argparse
import pdb
import tracemalloc

#Pyro-NN information:
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D

from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan, primitives_2d, primitives_3d
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d
from pyronn.ct_reconstruction.layers import projection_3d


"""
Precondition:
- geom_type: string that specified the geometric type (parallel, fanflat, or cone)
- volume_shape: shape of volume
- volume_spacing: spacing of volume
- angular_rangePI: angular range as PI term
- detector_shape: shape of detector
- detector_spacing: spacing of detector
- projection_number: number of projections
- source_detector_distance: detector distance from source
- source_isocenter_distance: detector isocenter distance
Postcondition:
- It returns a geometry type based on information above
"""
def chooseGeometry(geom_type, volume_shape, volume_spacing, angular_rangePI, detector_shape,detector_spacing, projection_number, source_detector_distance, source_isocenter_distance):
    if geom_type == 'parallel':
        geo = GeometryParallel2D(volume_shape,volume_spacing,detector_shape, detector_spacing,projection_number, angular_rangePI)
        geo.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
        return geo
    elif geom_type == 'fanflat':
        geo = GeometryFan2D(volume_shape,volume_spacing,detector_shape,detector_spacing,projection_number, angular_rangePI, source_detector_distance,source_isocenter_distance)
        geo.set_central_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
        return geo
    else:
        geo = GeometryCone3D(volume_shape,volume_spacing,detector_shape, detector_spacing, projection_number, angular_rangePI, source_detector_distance,source_isocenter_distance)
        geo.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geo)) 
        return geo
"""
Precondition: geometry for sinogram to be created from
Accepted types:
-   GeometryCone3D
-   GeometryFan2D
-   GeometryParallel2D
Postcondition: array representing sinogram
"""
def createSinogramFromGenericGeometry(phantom, geometry):
    if isinstance(geometry,GeometryCone3D):
        return generate_sinogram(phantom,  projection_3d.cone_projection3d, geometry)
    elif isinstance(geometry,GeometryFan2D):
        return generate_sinogram(phantom, projection_2d.fan_projection2d, geometry)
    else:
        return generate_sinogram(phantom, projection_2d.parallel_projection2d, geometry)

def selectPhantom(geometry, volume_shape):
    if(isinstance(geometry, GeometryCone3D)):
        return shepp_logan.shepp_logan_3d(volume_shape)
    else:
        return shepp_logan.shepp_logan_enhanced(volume_shape)

# Main executable, requires use of --in
def sinoExecute():

    #Parser set-up, as based on parser
    #Current standard line: 
    #python SynoGenerator.py  --in imgs/00000001_img.flt --out sinos

    #python SynoGenerator.py  --in imgs/00000001_img.flt --out sinos --geom fanflat --angularRangeDef 360 --source_detector_distance 1600 --source_isocenter_distance 800 --dx 0.065
    parser = argparse.ArgumentParser(description='')
    #File input commands:

#     parser.add_argument('--in', dest='infile', default='.', help='input file -- directory or single file')
#     parser.add_argument('--out', dest='outfile', default='.', help='output directory')

#     parser.add_argument('--dx', dest='dx', type = float, default = 1, help = "pixel size (cm)")               #DIS
#     parser.add_argument('--volume_size_definition', dest = 'volume_size', type = int, default = 512, help = "The volume size in Z, Y, X order")
    
#     parser.add_argument('--volume_spacing_definition', dest = 'volume_spacing_definition', type = float, default = 0.35, help = "Define space between volume spaces")
#     parser.add_argument('--angular_range_definition', dest = 'angular_range', type = int, default = 180, help = "Range of rotation of scan")
#     parser.add_argument('--detector_shape_definition', dest = 'detector_shape', type = int, default = 729, help = "Define space of volume")

#     parser.add_argument('--fan_angle', dest = 'fan_angle', type = float, default = 35.0, help = "Angle of fan")
#     parser.add_argument('--det_spacing_definition', dest = 'detector_spacing', type = float, default = 1.0, help = "Define space between detector spaces")
#     parser.add_argument('--number_of_project', dest = 'project_number', type = int, default = 900, help = "Define number of projections")
    
#     parser.add_argument('--geom', dest='geom',default='parallel',help='geometry (parallel, or fanflat (cone not supported))')
#     parser.add_argument('--source_detector_distance', dest='source_detecfor_distance', type = int, default = 1200, help = "Define the distance of the detector") #DSO+DOD
#     parser.add_argument('--source_isocenter_distance', dest='source_isocenter_distance', type = int, default = 750, help = "Distance of Isoceles")               #DIS

#     parser.add_argument('--lr', dest='learning_rateArg', type=float, default=1e-2, help='initial learning rate for adam')
#     parser.add_argument('--epoch', dest='num_epochsArg', type=int, default=1000000, help='# of epoch')
# #********************* Add option to choose strategy and specify sinos **************************************
#     parser.add_argument('--strat', dest='strategy_typeArg', type=str, default="SART", help = 'image evaluation strategy')
#     parser.add_argument('--sinos', dest='which_sinosArg', type=str, default="ONE", help="specify ONE for single image processing, ALL for sinogram construction")


    parser.add_argument('--in', dest='infile', default='.', help='input file -- directory or single file')
    parser.add_argument('--out', dest='outfile', default='.', help='output directory')

    #Variable definition
    parser.add_argument('--dx', dest='dx', type = float, default = 1, help = "pixel size (cm)")               #DIS
    parser.add_argument('--volume_size_definition', dest = 'volume_size', type = int, default = 512, help = "The volume size in Z, Y, X order")
    
    parser.add_argument('--volume_spacing_definition', dest = 'volSpacingArg', type = float, default = 1.0, help = "Define space between volume spaces")
    parser.add_argument('--angular_range_definition', dest = 'angular_range', type = int, default = 180, help = "Range of rotation of scan")

    parser.add_argument('--detector_shape_definition', dest = 'detector_shape', type = int, default = 729, help = "Define space of volume")
    parser.add_argument('--fan_angle', dest = 'fan_angle', type = float, default = 35.0, help = "Angle of fan")
    
    parser.add_argument('--det_spacing_definition', dest = 'detSpacingArg', type = float, default = 1.0, help = "Define space between detector spaces")
    parser.add_argument('--number_of_project', dest = 'projectNumb', type = int, default = 900, help = "Define number of projections")
    parser.add_argument('--geom', dest='geom',default='parallel',help='geometry (parallel, or fanflat (cone not supported))')
    
    parser.add_argument('--source_detector_distance', dest='sourceDetectorDistanceArg', type = int, default = 2000, help = "Define the distance of the detector") #DSO+DOD
    parser.add_argument('--source_isocenter_distance', dest='sourceIsocenterDistanceArg', type = int, default = 1000, help = "Distance of Isoceles")               #DIS
    parser.add_argument('--counts',dest='countsArg',type=float,default=1e6,help='count rate (to determine noise)')

    #get arguments from command line
    args = parser.parse_args()
    infile, outfile = args.infile, args.outfile

    dx = args.dx
    volume_size = args.volume_size
    volume_shape = [volume_size, volume_size]
    volume_spacing = [args.volSpacingArg,args.volSpacingArg]
    angular_range = args.angular_range
    angular_rangePI = np.radians(angular_range)
    detector_shape = args.detector_shape
    fan_angle = args.fan_angle
    detector_spacing = args.detSpacingArg 
    projection_number = args.projectNumb  
    geom_type = args.geom
    source_detector_distance = args.sourceDetectorDistanceArg 
    source_isocenter_distance = args.sourceIsocenterDistanceArg 
    counts = args.countsArg
    #Takes in image
    images = []
    outnames = []
    
    if os.path.isdir(infile):		#generate list of filenames from directory
        fnames = glob(infile + '/*.flt')
    else:							#single filename
        fnames = []
        fnames.append(infile)
    
    #read in images from list of filenames
    for name in fnames:
        
        img = np.fromfile(name,dtype='f')
        img = img.reshape(volume_size,volume_size)
        images.append(img)
        head, tail = os.path.split(name)      #get name of file for output
        head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_img.flt
        outnames.append(head) 
    if geom_type == 'fanflat':
        ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
        detector_spacing = 2 * source_detector_distance * ft / detector_shape  #width of one detector pixel, calculated based on fan angle
    
    #Create geometry
    geometry = chooseGeometry(geom_type, volume_shape, volume_spacing, angular_rangePI, detector_shape,detector_spacing, projection_number,source_detector_distance, source_isocenter_distance)
    

    #Set tensor for generation
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    #Start session
    # - generate cino

    #Save sino to outer file
    for img,name in zip(images,outnames):       # Loop currently only prints one because it
                                                # doesn't do anything with img input
        
        with tf.Session(config=config) as sess:
            # Create sinogram
        
            sinogram = createSinogramFromGenericGeometry(img,geometry)
        sinogram = sinogram*dx
        sinogram = counts * np.exp(-sinogram)   #exponentiate
        sinogram = np.random.poisson(sinogram)  #add noise
        #sinogram = np.amax(sinogram,1)          #guard against zero values
        sinogram = -np.log(sinogram/counts)     #return to log domain
        # Saves it
        fileout = outfile + "/" + name + '_sino.flt'
        sinogram = np.float32(sinogram)     # This may be disnecessary
        sinogram.tofile(fileout)
        pdb.set_trace()
        #**********save image as png**********
        max_pixel = np.amax(sinogram)
        img = (sinogram/max_pixel) * 255
        img = np.round(img)

        plt.figure(num=None, figsize=(90, 40), facecolor='w', edgecolor='k')
        plt.style.use('grayscale')
        plt.imshow(img, interpolation = 'nearest')
        png_outname = (fileout + '.png')
        plt.tight_layout()
        plt.savefig(png_outname)
        #**************************************

        plt.close()
        del max_pixel
        del sinogram

if __name__ == '__main__':
    sinoExecute()