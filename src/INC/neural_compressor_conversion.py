# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
INC QUANTIZATION model saving
"""
# pylint: disable=C0301 E0401 C0103 W0612 W0702 I1101 C0411 C0413
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from neural_compressor.experimental import Quantization, common
# from neural_compressor.experimental import Benchmark
tf.compat.v1.disable_eager_execution()
import warnings # noqa: 402
import logging  # noqa: 402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# Define the command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--modelpath',
                        type=str,
                        required=False,
                        default='./output/updated_model.pb',
                        help='Model path trained with tensorflow ".pb" file')
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./output/output',
                        help='default output quantized model will be save in ./output/output folder')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./deploy.yaml',
                        help='Yaml file for quantizing model, default is "./deploy.yaml"')

    paramters = parser.parse_args()
    FLAGS = parser.parse_args()
    model_path = FLAGS.modelpath
    config_path = FLAGS.config
    out_path = FLAGS.outpath


# Defing variable
INPUT_IMAGE_SIZE = (300, 300)
padding = "SAME"  # @param ['SAME', 'VALID' ]
NUM_OUTPUT_CLASSES = 2  # @param {type: "number"}
code_batch_size = 20  # @param {type: "number"}
val_batch_size = 2
learning_rate = 0.001  # @param {type: "number"}
abs_val_path = './data/chest_xray/test'


def create_path_list(abspath="None"):
    """ Creating path reading function """
    pathlist = []
    for (root, dirs, files) in os.walk(abspath):
        for subdir in dirs:
            for imgpath in os.listdir(os.path.join(abspath, subdir)):
                if imgpath.endswith('.jpeg'):
                    img_append = os.path.join(abspath, subdir, imgpath)
                    pathlist.append(img_append)

        break
    return pathlist


def read_image(batch_size=4, LAST_INDEX=2, pathlist=None):   # Creating function for reading images in batch size
    """defining function for reading image in batch size"""
    x_batch, y_batch = [], []
    for imagepath in pathlist[LAST_INDEX:LAST_INDEX+batch_size]:
        try:
            image = cv2.imread(imagepath)
            image = cv2.resize(image, dsize=INPUT_IMAGE_SIZE)
            image = image / 255.0
        except:  # noqa: E722
            logger.info("Please installed the required Opencv version")
        if imagepath.split('/')[-2] == 'NORMAL':
            y_var = np.array([0, 1])
        else:
            y_var = np.array(([1, 0]))
        x_batch.append(image)
        y_batch.append(y_var)
    x_batch_train = np.stack(x_batch, axis=0)
    y_batch_train = np.stack(y_batch, axis=0)
    return x_batch_train, y_batch_train


path_val_list = create_path_list(abs_val_path)


class Dataset():
    """Creating Dataset class for getting Image and labels"""
    def __init__(self):
        test_images, test_labels = read_image(100, 4, path_val_list)
        self.test_images = test_images.astype(np.float32) / 255.0
        self.labels = test_labels

    def __getitem__(self, index):
        return self.test_images[index], self.labels[index]

    def __len__(self):
        return len(self.test_images)


quantizer = Quantization()
quantizer.model = model_path
dataset = Dataset()
quantizer.calib_dataloader = common.DataLoader(dataset)
quantizer.fit()
q_model = quantizer.fit()
q_model.save(out_path)
