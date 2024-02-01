# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""System module."""
# pylint: disable= C0301,W0622,C0103,W0702,C0121,C0209,R0801,E0401 C0413 C0411
import os
import sys
from pathlib import Path
import warnings
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging  # noqa: E402

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


MODEL_PATH = './model'

try:
    if (os.path.isdir(MODEL_PATH) == False):  # noqa: E111 E712 E501   # pylint: disable=C0325
        logger.info(" paths not exist")  # noqa: E101 E117 W191
except:  # noqa: E722
    logger.info("path not found please get the data first")
    sys.exit()
else:
    print("Checking model files ===============")


model_meta_index = Path("./output/Medical_Diagnosis_CNN.meta") and Path("./output/Medical_Diagnosis_CNN.index")
if model_meta_index.exists():
    logger.info("Executing training notebook. This will take a while.........")
else:
    logger.info("Unable to execute")
    sys.exit()


#  The original freeze_graph function
#  from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.io.gfile.exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

# We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path


# We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/updated_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.io.gfile.GFile(output_graph, "wb") as fle:
            fle.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


# Pass the model directory path & Output node names through command prompt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./output/", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="Softmax", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)
