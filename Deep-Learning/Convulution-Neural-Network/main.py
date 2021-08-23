# We will using CIFAR-10 data set (from Canadian institute For Advanced Research) for
# classifying images across 10 categories

# ********************************** 1. Load the Data into Dictionaries **********************************

import os
import pickle

import keras.backend


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


entries = os.listdir("cifar-10-batches-py")
dir = "cifar-10-batches-py/"

all_data = {}

for i, entry in enumerate(entries):
    path = dir + str(entry)
    all_data[i] = unpickle(path)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[1]

# print(batch_meta)
# print(data_batch1.keys())


# ********************************** 2. Plot an Image **********************************

import matplotlib.pyplot as plt
import numpy as np

# Collect the data
X = data_batch1[b'data']
#         10000 img  32px RGB       (img 32 32 3)    uint8 type (0-255)
X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

# plt.imshow(X[0])
# plt.show()

# ********************************** 3. Helper function to handle data **********************************

from CifarHelper import CifarHelper

ch = CifarHelper(data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch)
ch.set_up_images()

# ********************************** 4. TensorFlow **********************************

import tensorflow as tf

x = keras.Input(dtype=tf.float32, shape=[None, 32, 32, 3])
y = keras.Input(dtype=tf.float32, shape=[None, 10])
hold_prob = keras.Input(dtype=tf.float32)

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)

