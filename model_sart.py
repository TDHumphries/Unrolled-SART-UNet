import time
import re
#from python_utils import *
from six.moves import xrange
import numpy as np
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import os

from utils import *
#Pyro-NN information:
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.layers import projection_2d
from pyronn.ct_reconstruction.layers import projection_3d
from pyronn.ct_reconstruction.layers import backprojection_2d

'''
Execute SART algorythmn, and build network
Precondition:
- sino: input sinogram
- numpix: 2d int array with size of image
- numbin: space of volume
- numtheta: number of rojections
- dx: number for rehorientation
- numits: number of iterations
- batch_size: int of size being passed
Postcondition: Created array D, and M, and x_initial, and eps(created bynp.finfo(float).eps )
'''
def SART_variables(sino, numpix, numbin, numtheta, geometry, dx, numits):
    eps = np.finfo(float).eps
    # Removes sess
    temp_D = tf.ones([1,numtheta, numbin])
    temp_M = tf.ones([1,numpix[0],numpix[0]])
    if isinstance(geometry,GeometryParallel2D):
        M = projection_2d.parallel_projection2d(temp_M, geometry)
        D = backprojection_2d.parallel_backprojection2d(temp_D, geometry)
    else:
        M = projection_2d.fan_projection2d(temp_M, geometry)
        D = backprojection_2d.fan_backprojection2d(temp_D, geometry)
    
    D = tf.maximum(tf.reshape(D,[numpix[0],numpix[0]]),eps)
    M = tf.maximum(tf.multiply(tf.reshape(M,[numtheta,numbin]),dx),eps)
    
    # x_initial = tf.zeros(len(numpix[0]))
    #x_initial = tf.convert_to_tensor(x_initial, np.float32)

    return D, M


'''
Execute LEARN algorythmn, and build network based in it
Precondition:
- sino: input sinogram
- geometry: 
- dx: number for rehorientation
- numpix: 2d int array with size of image
- numits: number of iterations
- D: ones array of backprojection
- M: ones array of projection
- current_x
- eps: eps from np.finfo(float).eps
- batch_size: int of size being passed
Postcondition: current_x is  the last tensorflow processed by the last layer
'''
def SART(sino, geometry, dx, numpix, numits, D, M, is_training):
    current_x = tf.zeros([1,numpix[0],numpix[0]])
    for iteration in range(numits):
        with tf.variable_scope('SART_layer{}'.format(iteration)):
            if isinstance(geometry,GeometryParallel2D):
                fp = projection_2d.parallel_projection2d(current_x, geometry)
            else:
                fp = projection_2d.fan_projection2d(current_x, geometry)
            fp = tf.reshape(fp,sino.shape)
            diff = tf.subtract(tf.multiply(fp,dx), sino, name="diff")
            tmp1 = tf.divide(diff, M, name="tmp1")
            if isinstance(geometry,GeometryParallel2D):
                bp = backprojection_2d.parallel_backprojection2d(tmp1, geometry)
            else:
                bp = backprojection_2d.fan_backprojection2d(tmp1, geometry)
            bp = tf.reshape(bp,current_x.shape)
            ind2 = tf.greater(tf.abs(bp), 1e3, name="ind2")
            zeros = tf.zeros_like(bp, name="zeros")
            bp = tf.where(ind2,zeros,bp, name="bp")
            tmp2 = tf.divide(bp, D, name="tmp2")

            # lam = tf.Variable(1, trainable=True, dtype=tf.float32, name='relaxation',
            #                   constraint=lambda x: tf.clip_by_value(x, 0, 2))
            lam = tf.Variable(1, trainable=True, dtype=tf.float32, name='relaxation')
            tmp2 = tf.multiply(lam, tmp2)

            current_x = tf.math.subtract(current_x, tmp2, name="current_x_minus_tmp2")
        current_x = cnn_layers(current_x, numpix, iteration, is_training)

    return current_x

'''
Execute LEARN algorythmn, and build network based in it
Precondition:
- sino: input sinogram
- numpix: 2d int array with size of image
- numits: number of iterations
- numtheta: volutme of shape
- geometry: geometry item passed in
- dx: number for rehorientation
- batch_size: int that is the batch size being trainned
Postcondition: current_x is  the last tensorflow processed by the last layer
'''
def LEARN(sino, geometry, dx, numpix, numits, is_training):    
    current_x = tf.zeros([1,numpix[0],numpix[0]])
    for iteration in range(numits):
        #pdb.set_trace()
        cnn_ret_val = cnn_layers(current_x, numpix, iteration, is_training)
        with tf.variable_scope('LEARN_layer{}'.format(iteration)):
            if isinstance(geometry,GeometryParallel2D):
                fp = projection_2d.parallel_projection2d(current_x, geometry)
            else:
                fp = projection_2d.fan_projection2d(current_x, geometry)
            fp = tf.reshape(fp,sino.shape)
            diff = tf.subtract(tf.multiply(fp,dx), sino, name="diff")
            # A^t (Ax^k-b)
            if isinstance(geometry,GeometryParallel2D):
                bp = backprojection_2d.parallel_backprojection2d(diff, geometry)
            else:
                bp = backprojection_2d.fan_backprojection2d(diff, geometry)
            bp = tf.reshape(bp,current_x.shape)
            bp = tf.multiply(bp, dx)
            lam = tf.Variable(0.01, trainable=True, dtype=tf.float32,name = 'stepsize',
                              constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
            grad = tf.multiply(lam,bp)

            current_x = tf.math.subtract(cnn_ret_val, grad)  #cnn_ret_val = current_x - regularization term
 
    return current_x

# ----------------------------------------------------------------------------------------------

'''
Picks correct strategy to create model, and executes it
Precondition:
- sino: input sinogram
- numpix: 2d int array with size of image
- numbin: space of volume
- numtheta: volutme of shape
- geometry: geometry item passed in
- dx: number for rehorientation
- numits: number of iterations as int
- strat: string labelein used strated as either 'SART' or 'LEARN'
- batch_size: int that is the batch size being trainned
- is_training: boolean specifying if it has to be trained (Default true)
- output_channels: specifies color of shape (Default 1 (grey))
Postcondition: Trained network based on given stategy
'''
def rncnn(sino, numpix, numbin, numtheta, geometry, dx, numits, strat, is_training):
    #Set tensor for generation
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    if(strat == 'SART'):
        D, M = SART_variables(sino, numpix, numbin, numtheta, geometry, dx, numits)
        return SART(sino, geometry, dx, numpix, numits, D, M, is_training)
        # D, M, x_initial, eps
    else:
        return LEARN(sino, geometry, dx, numpix, numits, is_training)

'''
Creates cnn_layer, trains based on information given, and returns possible tensor
Precondition: 
- user_input: tensor to be used for trainning
- numpix: 2d int array with size of image
- batch_size: size of batch as integer
- iteration: the int number that has the current iteration of the algorithmn
- is_training: boolean, true if trainning, false if not(default true)
- output_channels: number of chanels as int (default 1 (grey))
Postcondition: 
- returns user_input minus output from neural net
'''
def cnn_layers(user_input, numpix, iteration, is_training=True, output_channels=1):
    with tf.variable_scope('CNN_layer{}'.format(iteration)):
        user_input = tf.expand_dims(user_input,-1)      #add channel dimension
        output = tf.layers.conv2d(user_input, 64, 3, padding='same', activation=tf.nn.relu,name='layer_1')
        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='layer_2')
        # output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        # output = tf.layers.batch_normalization(output, training=is_training)
        output = tf.layers.conv2d(output, output_channels, 3, padding='same',name='layer_3')
        user_input = tf.math.subtract(user_input, output, name="Reunion")
    return tf.squeeze(user_input,-1) #get rid of channel dimension

'''
Class that takes in information, creates network, and offers support for other operations
'''
class ReconNet(object):
    
    '''
    Initializes the model if needed. Writtes information to tensorboard after
    Precondition:
    - sess: sess from tensorflow used for trainning
    - infile: file given to read information from
    - outfile: file to write to
    - x0_file: file with true image
    - xtrue_file
    - numpix: 2d int array with size of image
    - dx: number for rehorientation
    - numbin: space of volume
    - numtheta: volutme of shape
    - numits: number of iterations as int
    - geometry:  geometry object used for construction of proper projection/backrojection
    - num_epochs: number of epochs as int
    - strat_type: type  of strategy as string (either 'LEARN' or 'SART')
     - batch_size: size of batch as int(default: 128)
    '''
    def __init__(self, sess, numpix,dx,numbin,numtheta,numits,
                geometry, num_epochs, strat_type):

        self.cnn_model_name = "model_sart_1"
        self.logdir = "./logs/" + self.cnn_model_name

        self.numpix, self.dx, self.numbin, self.numtheta, self.numits = numpix, dx, numbin, numtheta, numits
        # self.true_sino = true_sino
        # self.noisy_sino = noisy_sino
        self.geometry = geometry
        self.num_epochs = num_epochs

        self.sess = sess
        # build model
        self.Y_ = tf.placeholder(tf.float32, shape=[1, numpix[0], numpix[1]], name='ndct_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, shape=[1,numtheta, numbin], name='svct_sino')
        self.Y = rncnn(self.X, numpix, numbin, numtheta, geometry, dx, numits, strat_type, self.is_training)
        # self.loss = (1.0 / batch_size) * tf.image.ssim(self.Y_, self.Y, range)
        self.loss = tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        # optimizer = tf.train.AdamOptimizer(self.lr).minimizer(-1 * self.loss)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # self.variables_names = [v.name for v in tf.trainable_variables()]
        # print(self.variables_names)
        regexp = re.compile(r'SART_layer\d{1,2}/relaxation:\d{1,2}')
        self.variables_lam = [v for v in tf.trainable_variables() if regexp.search(v.name)]
        print(self.variables_lam)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
        writer = tf.summary.FileWriter(self.logdir, self.sess.graph)


    '''
    Evaluates the model given information to evaluate it with
    Precondition: 
    - iter_num: number of iteration
    - ndct_test_data: array with ndct data to be tested
    - ldct_test_data: array with ldct data to be tested
    - sample_dir: directory to save samples
    - summary_merged: information important to the tensorboard
    - summary_writer: writter of summary
    - summ_img: Image saved for tensorboard
    '''
    def evaluate(self, epoch, iter_num, ndct_test_data,ldct_test_data, sample_dir, summary_merged, summary_writer, summ_img):

        f = open('archive_console/' + self.cnn_model_name, 'a+')

        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0      
        for idx in xrange(len(ldct_test_data)):
            noisy_image = np.expand_dims(ldct_test_data[idx],0)  #add batch dimension
            clean_image = np.expand_dims(ndct_test_data[idx],0)
            output_clean_image, psnr_summary, temp_img = self.sess.run(
                [self.Y, summary_merged, summ_img],
                feed_dict={self.X: noisy_image,
                           self.Y_: clean_image,
                           self.is_training: False})
            #pdb.set_trace()
            summary_writer.add_summary(psnr_summary, iter_num)
            summary_writer.add_summary(temp_img, iter_num)


            scalef= np.amax(clean_image)
            clean_image = np.clip(255 * clean_image/scalef, 0, 255).astype('uint8')
            output_clean_image = np.clip(255 * output_clean_image/scalef, 0, 255).astype('uint8')
            
            # calculate PSNR
            psnr = cal_psnr(clean_image, output_clean_image)
            f.write("img%d PSNR: %.2f" % (idx + 1, psnr) + "\n")
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            # print()
            # print("clean_image before saving as image: ",clean_image.shape)
            # print()
            # print("noisy_image before saving as image: ",noisy_image.shape)
            # print()
            #clean_image, noisy_image= arr2Img(clean_image, noisy_image)
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        clean_image, output_clean_image)
        avg_psnr = psnr_sum / len(ndct_test_data)
        f.write("[epoch: " + str(epoch) + " ]" + "--- Test ---- Average PSNR %.2f ---" % avg_psnr + "\n")
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
        f.close()

#    def denoise(self, data):
#        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
#                                                              feed_dict={self.Y_: data, self.is_training: False})
#        return output_clean_image, noisy_image, psnr


    '''
    trains model given ndct data and ldct data
    Precondition: 
    - ndct_data: ndct data as array
    - ldct_data: ldct data as array
    - ndct_eval_data: ndct data for  evaluation
    - ldct_eval_data: ldct data for evaluation
    - batch_size: size of batch as int
    - ckpt_dir: directory to save checkpoints
    - epoch: epoch number
    - lr: learning rate
    - sample_dir: directory to save samples a string
    - eval_every_epoch: estabilishes if every epoch is going to be evaluated. 1 if true, else it doesn't (default 1)
    '''
    def train(self, ndct_train_data, svct_train_data, ndct_test_data, svct_test_data, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        numBatch = int(ndct_train_data.shape[0] / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        tempY = tf.reshape(self.Y, [batch_size,self.numpix[0],self.numpix[1],1])
        img = tf.summary.image('denoised image', tempY, max_outputs=1)
        writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(start_epoch, iter_num, ndct_test_data, svct_test_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
        random_seed = np.random.randint(np.iinfo(np.uint32).max, size=epoch, dtype=np.uint32)      #different random seed for shuffling each time
        store_variables_lam = []
        store_loss = []
        for epoch in xrange(start_epoch, epoch):
            
            state = np.random.get_state()       #preserve state to avoid altering other random processes (e.g. optimization)
            np.random.seed(random_seed[epoch])  #set random seed for shuffling
            np.random.shuffle(ndct_train_data)
            np.random.seed(random_seed[epoch])
            np.random.shuffle(svct_train_data)
            np.random.set_state(state)          #return to previous state
            
#            p = np.random.permutation(len(ndct_train_data))
#            ndct_train_data, svct_train_data = ndct_train_data[p], svct_train_data[p]    #too memory intensive
            #pdb.set_trace()
            for batch_id in xrange(start_step, numBatch):
                ndct_batch_images, svct_batch_sinos = ndct_train_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :], svct_train_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :]

                # learning rate decay
                # _, Y, loss, summary = self.sess.run([self.train_op, self.Y, self.loss, merged],
                #                                  feed_dict={self.Y_: ndct_batch_images, self.X:svct_batch_sinos, self.lr: lr[epoch],
                #                                             self.is_training: True})

                # learning rate decay with variables_lam
                _, Y, loss, summary, variables_lam = self.sess.run(
                    [self.train_op, self.Y, self.loss, merged, self.variables_lam],
                    feed_dict={self.Y_: ndct_batch_images, self.X: svct_batch_sinos, self.lr: lr[epoch],
                               self.is_training: True})
                store_variables_lam.append(variables_lam)
                store_loss.append(loss)
                print("store_variables_lam shape", np.array(store_variables_lam).shape)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(epoch, iter_num, ndct_test_data, svct_test_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer, summ_img=img)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)

        store_variables_lam = np.array(store_variables_lam)
        np.save('archive_lam/' + self.cnn_model_name, store_variables_lam)
        store_loss = np.array(store_loss)
        np.save('archive_loss/' + self.cnn_model_name, store_loss)

        print("[*] Finish training.")

    '''
    Saves the model at given checkpoint
    Precondition: 
    - iter_num: mumber of iteration present
    - ckpt_dir: checkpoint directory used to save informatio in
    - model_name: name of model 
    '''
    def save(self, iter_num, ckpt_dir, model_name='rncnn-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    '''
    Loads model, if there is one existant
    Precondition:
    - checkpoint_dir: director with checkpoint to have model loaded from
    '''
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    '''
    Tests model
    Precondition: 
    - ldct_files: ldct data to be tested
    - ndct_files: ndct data to be tested
    - ckpt_dir: checkpoint directory 
    - save_dir: directory to save information
    '''
    def test(self, ldct_files, ndct_files, ckpt_dir, save_dir):
        """Test rncnn"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(ldct_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] start testing...")
        rawfiles= [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=idx)), 'wb') for idx in range (len(ndct_files))]
        for idx in xrange(len(ldct_files)):
            noisy_image= ldct_files[idx]
            clean_image= ndct_files[idx]
            output_clean_image = self.sess.run(
                [self.Y],
                feed_dict={self.X: noisy_image,
                           self.Y_: clean_image,
                           self.is_training: False})
            output_clean_image= np.asarray(output_clean_image)
            #output_clean_image= output_clean_image[255, :, :, :, :]
            #scalef= max(np.amax(clean_image), np.amax(output_clean_image))
            #noisy_image = np.clip(255 * noisy_image/scalef, 0, 255).astype('uint8')
            #scaled_output = np.clip(255 * output_clean_image/scalef, 0, 255).astype('uint8')
            #clean_image = np.clip(255 * clean_image/scalef, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            #output_clean_image, noisy_image= arr2Img(output_clean_image, noisy_image)
            #clean_image= np.reshape(clean_image, (512, 512))
            #clean_image= Image.fromarray(clean_image, 'L')
            #save_images(os.path.join(save_dir, 'test_%d.flt' % (idx + 1)),
            #            clean_image, noisy_image, output_clean_image)
            output_clean_image.tofile(rawfiles[idx])
        avg_psnr = psnr_sum / len(ndct_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
