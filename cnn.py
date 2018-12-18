import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import read_data


def rnn_conv_lstm_model(width, height, sequence_length, cluster_size):
    LR=1e-3
    convent = input_data( shape=[None,48,48,3],name='input')
    convent = conv_2d(convent, 32, 4, activation='relu')
    convent = max_pool_2d(convent, 4);
    convent = conv_2d(convent, 64, 5, activation='relu')
    convent = max_pool_2d(convent, 4)
    convent = conv_2d(convent, 128, 4, activation='relu')
    convent = max_pool_2d(convent, 4)
    convent = conv_2d(convent, 64, 4, activation='relu')
    convent = max_pool_2d(convent, 2)
    convent = conv_2d(convent, 32, 4, activation='relu')
    convent = max_pool_2d(convent, 2)
    convent = fully_connected(convent, n_units=1024,bias=.4, activation='relu');
    convent = dropout(convent, .1)
    convent = fully_connected(convent, cluster_size, activation='softmax')
    convent = regression(convent, optimizer='adam', loss='categorical_crossentropy',
                         learning_rate=LR, name='targets')
    return convent


def create_train_data():
    #this function get the data and handel the data_set shaping thin return the data_set and the length of the dic
    data_set, names = read_data.read();
    length, encode = read_data.generate_id(names)
    data_set = [[data_set[i][0], encode[data_set[i][1]]] for i in range(len(data_set))]
    #shuffle(data_set)
    #np.save('training_data.npy',data_set)
    return data_set,encode,length


def pars_data_to_x_y(data_set):
    x_train = np.array([data_set[i][0] for i in range(len(data_set))])
    y_train = np.array([data_set[i][1] for i in range(len(data_set))])
    return x_train, y_train


def start_model(X_train, Y_train, convent):
    model = tflearn.DNN(convent, tensorboard_dir='log', tensorboard_verbose=0)
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=3,
                        snapshot_step=300,validation_set=({'input': X_train}, {'targets': Y_train}),
                        show_metric=True)
    return model


def predict(model, imgs, threshold=.5):
    prediction = model.predict(imgs);
    prediction = [[int(i>threshold) for i in predicted] for predicted in prediction]
    return prediction



#data_set,dic, dic_length = create_train_data();
#length = len(data_set)
#train = data_set
#X_train, Y_train = pars_data_to_x_y(data_set)
#cluster_size =np.ceil(np.log2(dic_length))
#convent = rnn_conv_lstm_model(48,48,3,cluster_size)
#model = start_model(X_train,Y_train,convent)
#fig = plt.figure(figsize=(16,12))
#print(model.predict([X_train[1]]))
#print(predict(model,[X_train[5]]))
