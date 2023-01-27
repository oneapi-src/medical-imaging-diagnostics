# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Openvino quantization
"""

# pylint: disable= W0611 C0411 E0401 C0115 W1203
from pathlib import Path
import tensorflow as tf  # noqa: F401
import copy
import os
import sys  # noqa: F401
import urllib  # noqa: F401
import argparse
import cv2
#import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from addict import Dict
from compression.api import DataLoader, Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from openvino.inference_engine import IECore  # noqa: F401
#from pathlib import Path
from PIL import Image  # noqa: F401
import warnings
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=False, default=Path('/home/azureuser/oneAPI-MedicalDiagnosis-DL/data/chest_xray/val'), help='dataset path')  # noqa: E501
    parser.add_argument('--modelpath', type=str, required=False, default=Path("model/Medical_Diagnosis_CNN.xml"), help='Model path trained')   # noqa: E501
    FLAGS = parser.parse_args()
    model_xml = Path(FLAGS.modelpath)
    data_dir = Path(FLAGS.datapath)


if not model_xml.exists():
    logger.info("Not Executing training notebook....")


# defining model structure
model_config = Dict(
    {
        "model_name": "Medical_Diagnosis_CNN",
        "model": "model/Medical_Diagnosis_CNN.xml",
        "weights": "model/Medical_Diagnosis_CNN.bin",
    }
)

engine_config = Dict({"device": "CPU", "stat_requests_number": 2, "eval_requests_number": 2})

algorithms = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            "stat_subset_size": 1000,
        },
    }
]

# Defining all required function to run openvino model
class ClassificationDataLoader(DataLoader):
    """
    DataLoader for image data that is stored in a directory per category. For example, for
    categories _rose_ and _daisy_, rose images are expected in data_source/rose, daisy images
    in data_source/daisy.
    """

    def __init__(self, data_source):
        """
        :param data_source: path to data directory
        """
        self.data_source = Path(data_source)
        self.dataset = [p for p in data_dir.glob("**/*") if p.suffix in (".png", ".jpeg")]
        self.class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])

    def __len__(self):
        """
        Returns the number of elements in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get item from self.dataset at the specified index.
        Returns (annotation, image), where annotation is a tuple (index, class_index)
        and image a preprocessed image in network shape
        """
        if index >= len(self):
            raise IndexError
        filepath = self.dataset[index]
        annotation = (index, self.class_names.index(filepath.parent.name))
        image = self._read_image(filepath)
        return annotation, image

    def _read_image(self, index):
        """
        Read image at dataset[index] to memory, resize, convert to BGR and to network shape

        :param index: dataset index to read
        :return ndarray representation of image batch
        """
        image = cv2.imread(str(index))[:, :, (2, 1, 0)]
        image = cv2.resize(image, (300, 300)).astype(np.float32)
        return image.transpose(2, 0, 1)


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self._name = "accuracy"
        self._matches = []

    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        """
        Returns accuracy metric value for all model outputs. Results per image are stored in
        self._matches, where True means a correct prediction and False a wrong prediction.
        Accuracy is computed as the number of correct predictions divided by the total
        number of predictions.
        """
        num_correct = np.count_nonzero(self._matches)
        return {self._name: num_correct / len(self._matches)}

    def update(self, output, target):
        """Updates prediction matches.

        :param output: model output
        :param target: annotations
        """
        predict = np.argmax(output[0], axis=1)
        match = predict == target
        self._matches.append(match)

    def reset(self):
        """
        Resets the Accuracy metric. This is a required method that should initialize all
        attributes to their initial value.
        """
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-better", "type": "accuracy"}}

# Step 1: Load the model
model = load_model(model_config=model_config)
original_model = copy.deepcopy(model)

# Step 2: Initialize the data loader
data_loader = ClassificationDataLoader(data_source=data_dir)

# Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric
#        Compute metric results on original model
metric = Accuracy()

# Step 4: Initialize the engine for metric calculation and statistics collection
engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

# Step 5: Create a pipeline of compression algorithms
pipeline = create_pipeline(algo_config=algorithms, engine=engine)

# Step 6: Execute the pipeline
compressed_model = pipeline.run(model=model)

# Step 7 (Optional): Compress model weights quantized precision
#                    in order to reduce the size of final .bin file
compress_model_weights(model=compressed_model)

# Step 8: Save the compressed model and get the path to the model
compressed_model_paths = save_model(
    model=compressed_model, save_path=os.path.join(os.path.curdir, "model/optimized")
)
compressed_model_xml = Path(compressed_model_paths[0]["model"])
logger.info(f"The quantized model is stored in {compressed_model_xml}")

# Step 9 (Optional): Evaluate the original and compressed model. Print the results
original_metric_results = pipeline.evaluate(original_model)
if original_metric_results:
    print(f"Accuracy of the original model:  {next(iter(original_metric_results.values())):.5f}")
    logger.info("Accuracy of the original model: %f", {next(iter(original_metric_results.values()))})

quantized_metric_results = pipeline.evaluate(compressed_model)
if quantized_metric_results:
    print(f"Accuracy of the quantized model: {next(iter(quantized_metric_results.values())):.5f}")
    logger.info("Accuracy of the quantized model: {next(iter(quantized_metric_results.values()))}")
