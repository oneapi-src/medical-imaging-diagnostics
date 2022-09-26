# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
 initial model training.
"""
# pylint: disable=C0301 E0401 C0103 E0602 E0401 W0702 W0612 C0121 W0621 C0200 C0209 W0621 R0914 I1101 C0413 C0411

import os
import sys
import argparse
from pathlib import Path
import time
import warnings
from cv2 import cv2
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging  # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
tf1.get_logger().setLevel('INFO')
tf1.autograph.set_verbosity(1, alsologtostdout=True)


# Defing variable with predefined value
INPUT_IMAGE_SIZE = (300, 300)
padding = "SAME"  # @param ['SAME', 'VALID' ]
NUM_OUTPUT_CLASSES = 2  # @param {type: "number"}
code_batch_size = 20  # @param {type: "number"}
val_batch_size = 2
learning_rate = 0.001  # @param {type: "number"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data/chest_xray", help="Provide the exact data path")
    args = parser.parse_args()

data_dir_path = args.datadir
data_dir = Path("./data/chest_xray/train/NORMAL") and Path("./data/chest_xray/train/PNEUMONIA")
data_files = []
for p in data_dir.glob("**/*"):
    if p.suffix in (".jpeg"):  # pylint: disable=C0325
        data_files.append(p)
try:
    if(len(data_files) == 0):  # pylint: disable=C0325
        logger.info("unable to find Images")
except:  # noqa: E722
    logger.info('Images not found or format not supported,execution failed')
    sys.exit()  # noqa: F821


# Defining the creat path method where data needs to be read
def create_path_list(abspath="None"):
    ''' this is  function to get the path of images where it saved'''
    pathlist = []
    for (root, dirs, files) in os.walk(abspath):
        for subdir in dirs:
            for imgpath in os.listdir(os.path.join(abspath, subdir)):
                if imgpath.endswith('.jpeg'):
                    img_append = os.path.join(abspath, subdir, imgpath)
                    pathlist.append(img_append)
                    # print(img_append)
        break
    return pathlist


# Enter the paths of valid, training & testing just to make sure all the paths are correct
ABS_VAL_PATH = data_dir_path + "/val"
logger.info("ABS_VAL_PATH is ============================================>%s", ABS_VAL_PATH)
ABS_TRAIN_PATH = data_dir_path + "/train"
logger.info("ABS_TRAIN_PATH is ============================================>%s", ABS_TRAIN_PATH)
ABS_TEST_PATH = data_dir_path + "/test"
logger.info("ABS_TEST_PATH is ============================================>%s", ABS_TEST_PATH)

# Checking the train dataset
if (os.path.isdir(ABS_VAL_PATH) == True and os.path.isdir(ABS_TRAIN_PATH) == True and os.path.isdir(ABS_TEST_PATH) == True):  # noqa: E111 E712 E501
    logger.info("Data paths exist , executing the programme")  # noqa: E101 E117 W191
else:  # noqa:  E111
    logger.info("Valid path not found")  # noqa:  E117
    sys.exit()  # noqa: F821


# Read the image from the path defined above
def read_image(batch_size=4, LAST_INDEX=2, pathlist=None):
    '''This is the function where images will be read in batch_size'''
    x_batch, y_batch = [], []
    for imagepath in pathlist[LAST_INDEX:LAST_INDEX+batch_size]:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, dsize=INPUT_IMAGE_SIZE)
        image = image / 255.0
        if imagepath.split('/')[-2] == 'NORMAL':
            y = np.array([0, 1])
        else:
            y = np.array(([1, 0]))
        x_batch.append(image)
        y_batch.append(y)
    x_batch_train = np.stack(x_batch, axis=0)
    y_batch_train = np.stack(y_batch, axis=0)
    return x_batch_train, y_batch_train


leaky_relu_alpha = 0.2  # @param {type: "number"}  # CNN Training Module
dropout_rate = 0.5  # @param {type: "number"}


# Define the functions required running TF
def conv2d(inputs, filters, stride_size):
    ''' this is conv layer of the model'''
    out = tf.nn.conv2d(inputs, filters, strides=[1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.leaky_relu(out, alpha=leaky_relu_alpha)


def maxpool(inputs, pool_size, stride_size):
    '''this is the maxpool layer defination'''
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID', strides=[1, stride_size, stride_size, 1])


def dense(inputs, weights):
    '''this is the dense layer defination'''
    x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=leaky_relu_alpha)
    # return tf.nn.dropout(x, rate=dropout_rate)
    return x


output_classes = 2
initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name):

    '''this is get_weight function shape and name'''
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)


shapes = [
    [3, 3, 3, 16],
    [3, 3, 16, 16],
    [3, 3, 16, 32],
    [3, 3, 32, 32],
    [3, 3, 32, 64],
    [3, 3, 64, 64],
    [3, 3, 64, 128],
    [3, 3, 128, 128],
    [3, 3, 128, 256],
    [3, 3, 256, 256],
    [3, 3, 256, 512],
    [3, 3, 512, 512],
    [8192, 3600],
    [3600, 2400],
    [2400, 1600],
    [1600, 800],
    [800, 64],
    [64, output_classes],


]

weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i)))


#  defing the model function
def model(x):
    '''this is the model layer'''
    x = tf.cast(x, dtype=tf.float32)
    c_1 = conv2d(x, weights[0], stride_size=1)
    c_1 = conv2d(c_1, weights[1], stride_size=1)
    p_1 = maxpool(c_1, pool_size=2, stride_size=2)

    c_2 = conv2d(p_1, weights[2], stride_size=1)
    c_2 = conv2d(c_2, weights[3], stride_size=1)
    p_2 = maxpool(c_2, pool_size=2, stride_size=2)

    c_3 = conv2d(p_2, weights[4], stride_size=1)
    c_3 = conv2d(c_3, weights[5], stride_size=1)
    p_3 = maxpool(c_3, pool_size=2, stride_size=2)

    c_4 = conv2d(p_3, weights[6], stride_size=1)
    c_4 = conv2d(c_4, weights[7], stride_size=1)
    p_4 = maxpool(c_4, pool_size=2, stride_size=2)

    c_5 = conv2d(p_4, weights[8], stride_size=1)
    c_5 = conv2d(c_5, weights[9], stride_size=1)
    p_5 = maxpool(c_5, pool_size=2, stride_size=2)

    c_6 = conv2d(p_5, weights[10], stride_size=1)
    c_6 = conv2d(c_6, weights[11], stride_size=1)
    p_6 = maxpool(c_6, pool_size=2, stride_size=2)

    flatten = tf.reshape(p_6, shape=(tf.shape(p_6)[0], -1))

    d_1 = dense(flatten, weights[12])
    d_2 = dense(d_1, weights[13])
    d_3 = dense(d_2, weights[14])
    d_4 = dense(d_3, weights[15])
    d_5 = dense(d_4, weights[16])
    logits = tf.matmul(d_5, weights[17])

    return tf.nn.softmax(logits)


x = tf.placeholder(tf.float32, [None, 300, 300, 3])
y = tf.placeholder(tf.float32, [None, 2])
y_pred_1 = model(x)

# Loss Funstion usage
loss = tf1.losses.categorical_crossentropy(y, y_pred_1)   # define the loss function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
validation = optimizer.minimize(loss)

# Define the correct prediction during testing
correct_prediction = tf.equal(tf.argmax(y), tf.argmax(y_pred_1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
correct_prediction = tf.reduce_mean(correct_prediction)

# Initialisation
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8  # Set to number of physical cores
config.inter_op_parallelism_threads = 1  # Set to number of sockets
tf.Session(config=config)
s = tf.Session()
s.run(init)
start_time = time.time()

# Parsing the paths of Train, Test & Validation data to the create_path_list method
path_val_list = create_path_list(ABS_VAL_PATH)
path_train_list = create_path_list(ABS_TRAIN_PATH)
path_test_list = create_path_list(ABS_TEST_PATH)

# Intial warm up Training
for epoch in range(0, 4):
    logger.info("epoch in 4 we are evaluting warm up time : epoch_number -->%d ", epoch)
    LAST_INDEX = 0
    for step in range(0, int(len(path_val_list) / val_batch_size)):
        x_val_batch, y_val_batch = read_image(val_batch_size, LAST_INDEX, path_val_list)
        LAST_INDEX += val_batch_size
        s.run(validation, feed_dict={x: x_val_batch, y: y_val_batch})
logger.info("Warm Up  time in seconds --> %f", time.time() - start_time)
# Actual Training
trainingStart_time = time.time()
for epoch in range(0, 5):
    logger.info("epoch in 5 we are evaluting training time : epoch_number -->%d ", epoch)
    LAST_INDEX = 0
    for step in range(0, int(len(path_train_list) / code_batch_size)):
        x_batch, y_batch = read_image(code_batch_size, LAST_INDEX, path_train_list)
        LAST_INDEX += code_batch_size
        s.run(train, feed_dict={x: x_batch, y: y_batch})
logger.info("Total training time in seconds --> %f", time.time() - trainingStart_time)


# PREDICTION
LAST_INDEX = 0
BATCH_SIZE = 20
CORRECT_PREDICTION_COUNTER = 0
WRONG_PREDICTION_COUNTER = 0
LAST_INDEX = 0
for step in range(175, 375):
    LAST_INDEX = step
    x_batch, y_batch = read_image(BATCH_SIZE, LAST_INDEX, path_test_list)
    LAST_INDEX += BATCH_SIZE
    correct_prediction_temp = s.run(correct_prediction, feed_dict={x: x_batch, y: y_batch})
    # print(correct_prediction_temp)
    if correct_prediction_temp == 1.0:
        CORRECT_PREDICTION_COUNTER = CORRECT_PREDICTION_COUNTER + 1
    else:
        WRONG_PREDICTION_COUNTER = WRONG_PREDICTION_COUNTER+1
logger.info("the number of correct predcitions (TP + TN) is:%d", CORRECT_PREDICTION_COUNTER)
logger.info("The number of wrong predictions (FP + FN) is%d", WRONG_PREDICTION_COUNTER)

# Calculation of Accuracy
ACCURACY = CORRECT_PREDICTION_COUNTER / (CORRECT_PREDICTION_COUNTER + WRONG_PREDICTION_COUNTER)
logger.info("Accuracy of the model is :%f", (ACCURACY * 100))

# Saving the Model
model = tf.train.Saver()
model.save(s, './model/Medical_Diagnosis_CNN')
