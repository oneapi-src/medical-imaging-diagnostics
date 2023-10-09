# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
INC Validation
"""
# pylint: disable=C0301 E0401 C0103 W0612 W0702 E0401 I1101 C0411 C0413 W0105 W0611
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from neural_compressor.experimental import Quantization, common  # noqa: F401
from neural_compressor.experimental import Benchmark
tf.compat.v1.disable_eager_execution()
import warnings  # noqa: E402
import logging   # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # ## Parameters
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fp32modelpath', type=str, required=False, help='Model path trained with tensorflow ".pb" file')  # noqa: E501
    group.add_argument('--int8modelpath', type=str, required=False, help='load the quantized model folder')   # noqa: E501
    parser.add_argument('--datapath', type=str, required=False, default='./data/test', help='dataset path')  # noqa: E501
    parser.add_argument('--config', type=str, required=False, default='./deploy.yaml', help='Yaml file for quantizing model, default is "./deploy.yaml"')   # noqa: E501
    FLAGS = parser.parse_args()
    fp_model = FLAGS.fp32modelpath
    data_folder = FLAGS.datapath
    config_path = FLAGS.config
    int_model = FLAGS.int8modelpath   


# Defing variable
INPUT_IMAGE_SIZE = (300, 300)
padding = "SAME"  # @param ['SAME', 'VALID' ]
NUM_OUTPUT_CLASSES = 2  # @param {type: "number"}
code_batch_size = 20  # @param {type: "number"}
val_batch_size = 2
learning_rate = 0.001  # @param {type: "number"}
abs_val_path = data_folder


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


def read_image(batch_size=4, Last_index=2, pathlist=None):
    """defining function for reading image in batch size"""
    x_batch, y_batch = [], []
    for imagepath in pathlist[Last_index:Last_index+batch_size]:
        try:
            image = cv2.imread(imagepath)
            image = cv2.resize(image, dsize=INPUT_IMAGE_SIZE)
            image = image / 255.0
        except:  # noqa: E722
            print("Please installed the required Opencv version")
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

'''
# Quantization(As we have already quantized model so ignoring this part)
quantizer = Quantization()
quantizer.model = fp_model
dataset = Dataset()
quantizer.calib_dataloader = common.DataLoader(dataset)
quantizer.fit()
q_model = quantizer.fit()
q_model.save(int_model)
'''
dataset = Dataset()
if FLAGS.fp32modelpath is not None:
    logger.info("Evaluating the Normal Model=========================================================")
    evaluator = Benchmark(config_path)
    evaluator.model = fp_model
    evaluator.b_dataloader = common.DataLoader(dataset)
    #print(evaluator.model)
    logger.info(evaluator('performance'))

if FLAGS.int8modelpath is not None:
    logger.info("Evaluating the compressed Model=========================================================")
    evaluator = Benchmark(config_path)
    evaluator.model = int_model
    evaluator.b_dataloader = common.DataLoader(dataset)
    #print(evaluator.model)
    logger.info(evaluator('performance'))
