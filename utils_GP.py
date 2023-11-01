import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
dtype = np.float32

####################
###  DATALOADER  ###
####################

class DataGeneratorMIL(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=256, dim=(512,512,512), n_channels=3,
                 n_classes=2, shuffle=True, is_train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.is_train = (labels is not None) and is_train
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__data_generation(list_IDs_temp)
        # Generate data
        if self.is_train:
            y = self.labels[index*self.batch_size:(index+1)*self.batch_size]
            return np.array(X), np.array(y, dtype='float64')
        else:
            return np.array(X)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.is_train:
                ids = train_bags_dic[ID]
            else:
                ids = test_bags_dic[ID]
            imgs = []
            for idx in ids:
                if idx == 'all_zeros':
                    img = np.zeros((self.dim[0], self.dim[1], self.n_channels))
                    imgs.append(img)
                    continue
                if self.is_train:
                    _dir = train_files_loc[idx]
                    img = np.load(_dir + idx[:-4] + '.npy')
                    img = cv2.resize(img, (self.dim[1], self.dim[0]))
                    imgs.append(img)
                else:
                    img = np.load(test_images_dir + idx[:-4] + '.npy')
                    img = cv2.resize(img, (self.dim[1], self.dim[0]))
                    imgs.append(img)
            X[i,] = np.transpose(imgs, [1,2,0,3])
                
        return X

################
### CNN PART ###
################

# Convolutional layers 
Conv1 = layers.Conv2D(16, (5, 5), data_format="channels_last", activation='relu', kernel_initializer='glorot_uniform', padding='same')
Conv2 = layers.Conv2D(32, (3,3),  data_format="channels_last", activation='relu')
Conv3 = layers.Conv2D(32, (3,3),  data_format="channels_last", activation='relu')
Conv4 = layers.Conv2D(32, (3,3),  data_format="channels_last", activation='relu')
Conv5 = layers.Conv2D(32, (3,3),  data_format="channels_last", activation='relu')
Conv6 = layers.Conv2D(32, (3,3),  data_format="channels_last", activation='relu')

def dense(inp, n_neur, drop=0.2, act='relu'):
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(n_neur)(x)
    x = layers.Dropout(drop)(x)
    return layers.Activation(act)(x)

def VGG(inp):
    inp = tf.reshape(tf.transpose(inp, perm=(0,3,1,2,4)), shape=(-1,dim[0], dim[1], 3))
    x = Conv1(inp) 
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), data_format="channels_last", strides=(2, 2))(x)
    x = Conv2(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2), data_format="channels_last")(x)
    x = layers.Dropout(0.3)(x)

    x = Conv3(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2),  data_format="channels_last")(x)
    x = Conv4(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2),  data_format="channels_last")(x)

    x = Conv5(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2),  data_format="channels_last")(x)
    x = layers.Dropout(0.3)(x)

    x = Conv6(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), strides=(2, 2),  data_format="channels_last")(x)
    x = layers.Dropout(0.3)(x)

    return x

def attention(inp):
    out = layers.Dense(D, activation='tanh')(inp)
    out = layers.Dense(1, use_bias = False)(out)   # change
    out = tf.reshape(out, shape=(-1,dim[2]))
    return tf.nn.softmax(out, axis=1)

def CNN_part():
    inp = keras.Input(shape=(*dim,3))
    out = VGG(inp)
    return keras.Model(inputs=inp, outputs=out)

def attention_ATT_GP(dim_):
    inp = keras.Input(shape=dim_)
    H = layers.Flatten()(inp)
    A = attention(H)
    H = tf.reshape(H, shape=(-1,dim[2], H.shape[1]))
    A = tf.expand_dims(A, axis=1)
    intermediate = tf.linalg.matmul(A,H)
    intermediate = tf.squeeze(intermediate, axis=1)
    out = layers.Dense(8)(intermediate)
    return keras.Model(inputs=inp, outputs=out)

def attention_GP_ATT(dim_, fc):
     inp = keras.Input(shape=dim_)
     H = inp
     if fc:
         H = layers.Flatten()(H)
         H = layers.Dense(8)(H)
     A = attention(H)
     H = tf.reshape(H, shape=(-1,dim[2], H.shape[1]))
     A = tf.expand_dims(A, axis=1)
     intermediate = tf.linalg.matmul(A,H)
     out = tf.squeeze(intermediate, axis=1)
     return keras.Model(inputs=inp, outputs=out)
    
####################
###  RBF KERNEL  ###
####################
class RBFKernelFn(tf.keras.layers.Layer):
    """
    RGF kernel for Gaussian processes.
    """
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', np.float32)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(1.0 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(1.0 * self._length_scale) # 5.
        )

###################
###   KL LOSS   ###
###################
def kl_loss(head, batch_size, kl_weight):
    def _kl_loss():
        num_training_points = head.variables[7]
        kl_weight = tf.cast(0.001 * batch_size / num_training_points, tf.float32)
        weight = tf.cast(kl_weight * batch_size / num_training_points, tf.float32)
        kl_div = tf.reduce_sum(head.layers[0].submodules[5].surrogate_posterior_kl_divergence_prior())

        loss = tf.multiply(weight, kl_div)
        # tf.print('kl_weight: ', kl_weight)
        tf.print('kl_loss: ', loss)
        # tf.print('u_var: ', head.variables[4])
        return loss

    return _kl_loss

######################
### EARLY STOPPING ###
######################

class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, metrics=[], patience=0, modes=[]):
        super(CustomEarlyStopping, self).__init__()
        self.metrics = metrics
        self.patience = patience
        self.modes = modes
        self.best_weights = None
        if len(self.modes) != len(self.metrics):
            raise AttributeError('Modes and metrics does not have the same length')
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_v = np.empty(len(self.metrics))
        for i in range(len(self.metrics)):
            if self.modes[i] == 'min':
                self.best_v[i] = np.Inf
            elif self.modes[i] == 'max':
                self.best_v[i] = -np.Inf
            else:
                raise ValueError('Mode is not valid')

    def on_epoch_end(self, epoch, logs=None): 
        best_v = np.empty(len(self.metrics))
        for i in range(len(self.metrics)):
            best_v[i] = logs.get(self.metrics[i])

        # If EVERY metric does not improve for 'patience' epochs, stop training early.
        improved = False
        for i in range(len(self.metrics)):
            if self.modes[i] == 'min':
                improved = improved or np.less(best_v[i], self.best_v[i])
            elif self.modes[i] == 'max':
                improved = improved or np.greater(best_v[i], self.best_v[i])
            else:
                raise ValueError('Mode is not valid')
                
        if improved:
            for i in range(len(self.metrics)):
                self.best_v[i] = best_v[i]
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
