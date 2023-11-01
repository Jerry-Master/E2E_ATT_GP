####################
###  LIBRARIES  ####
####################
import numpy as np
import pandas as pd
import os
from math import inf

# Neural network libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import random

import gpflow
import gpflux

from gpflow.config import default_float
# Remove tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Default float must be same type or it will give errors from time to time
gpflow.config.set_default_float("float32")
tf.keras.backend.set_floatx("float32")

import utils_GP as utils
from utils_GP import DataGeneratorMIL, CNN_part, attention_ATT_GP
from GPLayerSeq import GPLayerSeq

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

####################
### DATA LOADING ###
####################

print('Starting preprocessing of bags')
test_images_dir = '.../Data/test/'
utils.test_images_dir = test_images_dir

train_images_dir1 = '.../Data/train1/'
train_images_dir2 = '.../Data/train2/'
train_images_dir3 = '.../Data/train3/'

test_files = os.listdir(test_images_dir)
train_files1 = set(os.listdir(train_images_dir1))
train_files2 = set(os.listdir(train_images_dir2))
train_files3 = set(os.listdir(train_images_dir3))

train_bags = pd.read_csv(".../Data/bags_train.csv")
test_bags = pd.read_csv(".../Data/bags_test.csv")

dirs_ = [train_images_dir1, train_images_dir2, train_images_dir3]

train_files_loc = {k: dirs_[(k[:-4]+'.npy' in train_files1) * 1 \
                            + (k[:-4]+'.npy' in train_files2) * 2 \
                            + (k[:-4]+'.npy' in train_files3) * 3 - 1] for k in train_bags.instance_name}
utils.train_files_loc = train_files_loc

train_files1_dcm = [k[:-4] + '.dcm' for k in train_files1]
train_files2_dcm = [k[:-4] + '.dcm' for k in train_files2]
train_files3_dcm = [k[:-4] + '.dcm' for k in train_files3]
train_bags = train_bags[train_bags.instance_name.isin(train_files1_dcm) | train_bags.instance_name.isin(train_files2_dcm) | train_bags.instance_name.isin(train_files3_dcm)]

test_files_dcm = [k[:-4] + '.dcm' for k in test_files]
test_bags = test_bags[test_bags.instance_name.isin(test_files_dcm)]

##########################
### BAGS PREPROCESSING ###
##########################

bag_size=57
utils.bag_size=bag_size

# # Train
added_train_bags = pd.DataFrame()
for idx in train_bags.bag_name.unique():
     bags = train_bags[train_bags.bag_name==idx].copy()
     num_add = bag_size - len(bags.instance_name)

     aux = bags.iloc[0].copy()
     aux.instance_label = 0
     aux.instance_name = 'all_zeros'
     for i in range(num_add):
         added_train_bags = added_train_bags.append(aux)

train_bags = train_bags.append(added_train_bags)

added_test_bags = pd.DataFrame()
for idx in test_bags.bag_name.unique():
     bags = test_bags[test_bags.bag_name==idx].copy()
     num_add = bag_size - len(bags.instance_name)

     aux = bags.iloc[0].copy()
     aux.instance_label = 0
     aux.instance_name = 'all_zeros'
     for i in range(num_add):
         added_test_bags = added_test_bags.append(aux)

test_bags = test_bags.append(added_test_bags)

# Using dictionaries instead of DataFrames to optimize
train_bags_dic = {k:list(train_bags[train_bags.bag_name==k].instance_name) for k in train_bags.bag_name.unique()}
test_bags_dic = {k:list(test_bags[test_bags.bag_name==k].instance_name) for k in test_bags.bag_name.unique()}

utils.train_bags_dic = train_bags_dic
utils.test_bags_dic = test_bags_dic

#######################
### HYPERPARAMETERS ###
#######################
print('Starting model')
strategy = tf.distribute.MirroredStrategy()

from tensorflow.python.client import device_lib
print('Devices available:', strategy.num_replicas_in_sync)

with strategy.scope():

    dim=(512,512,bag_size)
    utils.dim=dim
    batch_size=16
    num_data = batch_size
    num_inducing = 50  # flexible here
    output_dim = 8
    num_latent = 8
    D = 50
    utils.D = D

    def compute_inducing(X):
        X = cnn_part(X)
        return np.repeat(np.linspace(tf.math.reduce_min(X), tf.math.reduce_max(X), num_inducing).reshape(-1,1), num_latent,axis=1)

    def compute_min_max(X):
        X = cnn_part(X)
        return (tf.math.reduce_min(X), tf.math.reduce_max(X))
    
    def create_inducing_points(train_dataset2):
        aggregation = 'MIN_MAX'
        start = 0
        ind_bags = 16
        if aggregation == "MEAN":
            ind_points = np.zeros((num_inducing, num_latent))
            for i in range(ind_bags):
                ind_points += compute_inducing(train_dataset2.__getitem__(start+i)[0])
            ind_points /= ind_bags
        else: # MIN_MAX
            min_ = inf
            max_ = -inf
            for i in range(ind_bags):
                min_max = compute_min_max(train_dataset2.__getitem__(start+i)[0])
                min_ = min(min_, min_max[0])
                max_ = max(max_, min_max[1])
            ind_points = np.repeat(np.linspace(min_, max_, num_inducing).reshape(-1,1), num_latent,axis=1)

        inducing_variable = gpflow.inducing_variables.InducingPoints(ind_points)
        return inducing_variable

#####################################
###   TRAINING AND TESTING LOOP   ###
#####################################

scale_factor = 0.5    # Flexible
scale_factor = scale_factor / (1 - scale_factor)

bags = train_bags.groupby('bag_name').max()
bags2 = test_bags.groupby('bag_name').max()
test_dataset = DataGeneratorMIL(np.array(bags2.index), bags2.bag_label, batch_size=1, dim=dim, is_train=False)
y_test = bags2.bag_label

X_train, X_val, y_train, y_val = train_test_split(np.array(bags.index), bags.bag_label,
                                                    test_size=0.25,
                                                    stratify=bags.bag_label) 

train_dataset = DataGeneratorMIL(X_train, y_train, batch_size=batch_size, dim=dim)
val_dataset = DataGeneratorMIL(X_val, y_val, batch_size=batch_size, dim=dim)
train_dataset2 = DataGeneratorMIL(X_train, y_train, batch_size=1, dim=dim)

with strategy.scope():
    cnn_part = CNN_part()
    att_part = attention_ATT_GP((dim[0]//64-2, dim[1]//64-2, 32))

    kernel = gpflow.kernels.SquaredExponential(variance=0.5, lengthscales=[1.5])
    inducing_variable = create_inducing_points(train_dataset2)
    gp_layer = GPLayerSeq(
        kernel, inducing_variable, num_data=num_data * dim[2], num_latent_gps=output_dim, 
        mean_function=gpflow.mean_functions.Identity(), scale_factor=scale_factor
    )

    model = keras.Sequential([cnn_part,
                              att_part,
                              gp_layer,
                              layers.Dense(1, activation='sigmoid')])
    auc = tf.keras.metrics.AUC(name='auc')
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
    model.compile(
        optimizer=adam, 
        loss='binary_crossentropy',
        metrics=[auc, 'accuracy']   
        )

checkpoint_path = ".../Model_NAME.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
Model_Train = False

if Model_Train is True:
    # Create a callback that saves the model's weights
    monitor = 'val_auc'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor=monitor,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1, mode='max')
    earlyStopping = EarlyStopping(monitor=monitor, patience=8, verbose=1, mode='max')

    print('Starting training on (AUC):')
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=200, callbacks=[earlyStopping, cp_callback])
    with open('.../trainHistoryDict_auc', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Retrain the model with early stopping based on accuracy
    monitor = 'val_accuracy'
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor=monitor,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                verbose=1, mode='max')
    earlyStopping = EarlyStopping(monitor=monitor, patience=8, verbose=1, mode='max')

    print('Starting training on (Accuracy):')
    model.load_weights(checkpoint_path)
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=200, callbacks=[earlyStopping, cp_callback])
    with open('.../trainHistoryDict_accuracy', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
else:
    # Testing after training the model
    with strategy.scope():
        model.load_weights(checkpoint_path)
        preds = model.predict(test_dataset)
        auc_score = roc_auc_score(y_test, preds[:,0])
        preds_value = (preds[:,0] > 0.5) * 1
        acc_score = accuracy_score(y_test, preds_value)
        prec_score = precision_score(y_test, preds_value)
        rec_score = recall_score(y_test, preds_value)
        F1_score = f1_score(y_test, preds_value)

    with open('.../scores.txt', 'a') as scores_file:
        scores_file.write('    AUC: ' + str(auc_score) + '\n')
        scores_file.write('    Accuracy: ' + str(acc_score) + '\n')
        scores_file.write('    Precision: ' + str(prec_score) + '\n')
        scores_file.write('    Recall: ' + str(rec_score) + '\n')
        scores_file.write('    F1 score: ' + str(F1_score) + '\n')

