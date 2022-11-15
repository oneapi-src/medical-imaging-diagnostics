## **Application of AI in image-base abnormalities for different diseases classification using TensorFlow**

## **Table of Contents**
 - [Purpose](#purpose)
 - [Reference Solution](#reference-solution)
 - [Reference Implementation](#reference-implementation)
 - [Intel® Optimized Implementation](#optimizing-the-e2e-solution-with-intel%C2%AE-oneapi-components)
 - [Performance Observations](#performance-observations)

 <!-- Purpose -->
## Purpose

Medical diagnosis of image-base abnormalities for different diseases classification is the process of determining the abnormality or condition explains a person's symptoms and signs. It is most often referred to as diagnosis with the medical context being implicit.
Images are a significant component of the patient’s electronic healthcare record (EHR) and are one of the most challenging data sources to analyze as they are unstructured. As the number of images that require analysis and reporting per patient is growing, global concerns around shortages of radiologists have also been reported. AI-enabled diagnostic imaging aid can help address the challenge by increasing productivity, improving diagnosis and reading accuracy (e.g., reducing missed findings or false negatives), improving departmental throughput, and helping to reduce clinician burnout.


The most common and widely adopted application of AI algorithms in medical image diagnosis is in the classification of abnormalities. With the use of machine learning (ML) and deep learning, the AI algorithm identifies images within a study  that warrants further attention by the radiologist/reader to classify the diseases. This aids in reducing the read time as it draws the reader’s attention to the specific image and identifies abnormalities 

## Reference Solution

■ X-ray images are critical in the detection of lung cancer, pneumonia, certain tumors, abnormal masses, calcifications, etc. In this reference kit, we demonstrate the detection of pneumonia using X-ray images and how a CNN model architecture can help identify and localize pneumonia in chest X-ray (CXR) images.

■ The experiment aims to classify pneumonia x-ray images to detect abnormalities from the normal lung images. The goal to improve latency, throughput (Frames/sec), and accuracy of the abnormality detection by training a CNN model in batch and inference in real-time. Hyperparameter tuning is applied at training for further optimization. <br>

■ Since GPUs are the natural choice for deep learning and AI processing to achieve a higher FPS rate but they are also very expensive and memory consuming, the experiment applies model quantification to speed up the process using CPU, whilst reaching the standard FPS for these types of applications to operate, to show a more cost-effective option using Intel’s technology. When it comes to the deployment of this model on edge devices, with less computing and memory resources, the experiment applies further quantification and compression to the model whilst keeping the same level of accuracy showing a more efficient utilization of underlying computing resources <br>

## **Key Implementation Details**

■ In this refkit we highlighted the difference of using Intel OneAPI packages specially TensorFlow against the packages that of stock version of the same packages.<br>

■ In this refkit we use a CNN model architecture for image classification based on a dataset form healthcare domain.The CNN-based model is a promising method to diagnose the disease through X-ray images.
The time required for training the model, inference time and the accuracy of the model are captured for multiple runs on the stock version as well on the Intel OneAPI version. The average of these runs are considered and the comparison have been provided.

■  Model has been quantized using Intel® Neural Compressor and Intel® Distribution of OpenVINO™ Toolkit, which has shown high performance vectorized operations on Intel platforms

## Reference Implementation

### ***E2E Architecture***
### **Use Case E2E flow**
![Use_case_flow](assets/E2E_2.PNG)

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| X-ray Imaged data (Normal and Infected)          |  Disease classification

**Example Input**                                 | **Example Output** |
| :---: | :---: |
| <b>X ray Imaged data based on patient's complain <br></b> Fast breathing, shallow breathing, shortness of breath, or wheezing, Patient reports pain in throat, chest pain. fever and loss of appetite over  the last few days. | {'Normal': 0.1, 'Pneumonia ': 0.99}

### Reference Sources
*DataSet*:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia<br>

*Case Study & Repo*: https://becominghuman.ai/image-classification-with-tensorflow-2-0-without-keras-e6534adddab2

### Notes
***Please see this data set's applicable license for terms and conditions. Intel®Corporation does not own the rights to this data set and does not confer any rights to it.***

### Repository clone and Anaconda installation

```
git clone https://github.com/oneapi-src/medical-imaging-diagnostics.git
```

>**Note**: If you beginning to explore the reference kits on client machines such as a windows laptop, go to the [Running on Windows](#running-on-windows) section to ensure you are all set and come back here

>**Note**: The performance measurements were captured on Xeon based processors. The instructions will work on WSL, however some portions of the ref kits may run slower on a client machine, so utilize the flags supported to modify the epochs/batch size to run the training or inference faster. Additionally performance claims reported may not be seen on a windows based client machine. 

>**Note**: In this reference kit implementation already provides the necessary conda environment configurations to setup the software requirements. To utilize these environment scripts, first install Anaconda/Miniconda by following the instructions at the following link<br>[Anaconda installation](https://docs.anaconda.com/anaconda/install/linux/)

## Overview
### ***Software Requirements***
| **Package**                | **Stock Python**                
| :---                       | :---                            
| Opencv-python              | opencv-python=4.5.5.64          
| Numpy                      | numpy=1.22                    
| TensorFlow                 | tensorflow =2.8.0                
| neural-compressor          | NA                              
| openvino-dev               | NA                              

### Environment

Below are the developer environment used for this module on Azure. All the observations captured are based on these environment setup.

**Size** | **CPU Cores** | **Memory**  | **Intel CPU Family**
| :--- | :--: | :--: | :--:
| *Standard_D8_Vs5* | 8 | 32GB | ICELAKE

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
| `env/stock/stock-tf.yml`             | `stock-tf` | Python=3.8.x with Stock TensorFlow 2.8.0

### Dataset

| **Use case** | Automated methods to detect and classify Pneumonia diseases from medical images
| :--- | :---
| **Object of interest** | Medical diagnosis healthcare Industry
| **Size** | Total 5856 Images Pneumonia and Normal <br>
| **Train: Test Split** | 90:10

## Usage and Instructions

Below are the steps to reproduce the benchmarking results given in this repository
1. Environment Creation
2. Dataset preparation
3. Training CNN model
4. Hyperparameter tuning
5. Model Inference


### 1. Environment Creation

**Setting up the environment for Stock TensorFlow**<br>Follow the below conda installation commands to setup the Stock TensorFlow environment for the model training and prediction. 

```sh
conda env create -f env/stock/stock-tf.yml
```
*Activate stock conda environment*
Use the following command to activate the environment that was created:

```sh
conda activate stock-tf
```
### 2. Data preparation

> Chest X-Ray Images (Pneumonia) is downloaded and prepared by extracted in a  <b>data <b> folder before running the training python module.

```
Data downloading steps: 

1. Create a data folder using "mkdir data" and navigate to inside that folder

2. wget https://s3.eu-central-1.amazonaws.com/public.unit8.co/data/chest_xray.tar.gz

3. tar -xf chest_xray.tar.gz
 
```
 >**Note**: Make sure "chest_xray" folder should be inside "data" folder  
 scripts have been written in such folder structure
 
<br>Folder structure Looks as below after extraction of dataset.</br>
```
- data
    - chest_xray
        - train
            - NORMAL
            - PNEUMONIA
        - test
            - NORMAL
            - PNEUMONIA
        - val
            - NORMAL
            - PNEUMONIA
```

### 3. Training

We trained 6 convolution layers and 5 dense layers CNN architecture model to classify the normal and pneumonia from the production pipeline.

| **Input Size** | 416x608
| :--- | :---
| **Output Model format** | TensorFlow checkpoint

### Training CNN model

**Capturing the time for training**
<br>Run the training module as given below to start training and prediction using the active environment. This module takes option to run the training.
```
usage: medical_diagnosis_initial_training.py  [--datadir] 

optional arguments:
  -h,                   show this help message and exit
  
  --data_dir 
                        Absolute path to the data folder containing
                        "chest_xray" and "chest_xray" folder containing "train" "test" and "val" 
                         and each subfolders contain "Pneumonia" and "NORMAL" folders 
```

**Command to run training**

```sh
python src/medical_diagnosis_initial_training.py  --datadir ./data/chest_xray
```
By default, model checkpoint will be saved in "model" folder.

> **Note**: If any CV2 dependency comes like "cv2 import *ImportError: libGL.so.1: cannot open shared object file" please execute sudo apt install libgl1-mesa-glx

### 4. Hyperparameter tuning

 **hyperparameters used here are as below** 
<br> Dataset remains same with 90:10 split for Training and testing. It needs to be ran multiple times on the same dataset, across different hyper-parameters

Below parameters been used for tuning

<br>"learning rates"      : [0.001, 0.01]
<br>"batchsize"           : [10 ,20]

```
usage: medical_diagnosis_hyperparameter_tuning.py 

optional arguments:
  -h,                   show this help message and exit
  

  --data_dir 
                        Absolute path to the data folder containing
                        "chest_xray" and "chest_xray" folder containing "train" "test" and "val" 
                         and each subfolders contain "Pneumonia" and "NORMAL" folders

```
**Command to run hyperparameter tuning**

```sh
python src/medical_diagnosis_hyperparameter_tuning.py   --datadir  ./data/chest_xray
```
By default, best model checkpoint will be saved in "model" folder.

**Convert the model to frozen graph**

run the conversion module to convert the TensorFlow checkpoint model format to frozen graph format. 

```
usage: python src/model_conversion.py [-h] [--model_dir] [--output_node_names]

optional arguments:
  -h  
                            show this help message and exit
  --model_dir
                            Please provide the Latest Checkpoint path e.g for
                            "./model"...Default path is mentioned

  --output_node_names       Default path is mentioned as "Softmax"
```
**Command to run conversion**

```sh
python src/model_conversion.py --model_dir ./model  --output_node_names Softmax
```
>**Note** : Also we need to generate Stock frozen_graph.pb and move all stock model files in new folder named "stockmodel" inside model folder to avoid the overwrite model file conflict when we run scripts in Intel.

### 5. Inference

 Running inference using Stock TensorFlow using 2.8.0 

```
usage: inference.py [--codebatchsize ] [--modeldir ]

optional arguments:
  -h,                       show this help message and exit

  --codebatchsize           --codebatchsize
                              batchsize used for inference
                        
  --modeldir                --modeldir         
                              provide frozen Model path ".pb" file...users can also
                              use INC INT8 quantized model here

```
**Command to run inference**

```sh
python src/inference.py --codebatchsize 1  --modeldir ./stockmodel/updated_model.pb
```
>**Note** : As we mentioned earlier all the stock generated model need to be moved stockmodel folder and codebatchsize can be changed (1,32,64,128).

## Optimizing the E2E solution with Intel® oneAPI components

### **Use Case E2E flow**

![Use_case_flow](assets/E2E_1.PNG)

### **Optimized software components**
| **Package**                | **Intel Python**
| :---                       | :---
| Opencv-python              | opencv-python=4.5.5.64
| Numpy                      | numpy=1.22
| TensorFlow                 |tensorflow=2.9.0
| neural-compressor          | neural-compressor==1.12.0
| openvino-dev               | openvino-dev[tensorflow]==2022.1.0.dev20220316

### **Optimized Solution setup**

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
`env/intel/intel-tf.yml`             | `intel-tf` | Python=3.8.x with  TensorFlow 2.9.0 with OneDNN|

## Usage and Instructions
Below are the steps to reproduce the benchmarking results given in this repository
1. Environment Creation
2. Training CNN model
3. Hyperparameter tuning
4. Model Inference
5. Quantize trained models using INC and benchmarking
6. Quantize trained models using  Intel® Distribution of OpenVINO™ and benchmarking

### 1. Environment Creation

**Setting up the environment for Intel oneDNN optimized TensorFlow**<br>Follow the below conda installation commands to setup the Intel oneDNN optimized TensorFlow environment for the model training and prediction.
```sh
conda env create -f env/intel/intel-tf.yml
```
*Activate intel conda environment*
Use the following command to activate the environment that was created:

```sh
conda activate intel-tf 
export TF_ENABLE_ONEDNN_OPTS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```
>**Note**: We need to set the above flags everytime before running the scripts below

### 2. Training CNN model

**Capturing the time for training **
<br>Run the training module as given below to start training and prediction using the active environment. This module takes option to run the training.
```
usage: medical_diagnosis_initial_training.py  [--datadir] 

optional arguments:
  -h,                   show this help message and exit
  
  --data_dir 
                        Absolute path to the data folder containing
                        "chest_xray" and "chest_xray" folder containing "train" "test" and "val" 
                         and each subfolders contain "Pneumonia" and "NORMAL" folders 
```
**Command to run training**

```sh
python src/medical_diagnosis_initial_training.py  --datadir ./data/chest_xray
```
By default, model checkpoint will be saved in "model" folder.

> **Note**:  If any gcc dependency comes please upgrade it using sudo apt install build-essential.
Above training command will run in intel environment and the output trained model would be saved in TensorFlow checkpoint format.


### 3. Hyperparameter tuning

**hyperparameters used here are as below** 
<br> Dataset remains same with 90:10 split for Training and testing. It needs to be ran multiple times on the same dataset, across different hyper-parameters

Below parameters been used for tuning

<br>"learning rates"      : [0.001, 0.01]
<br>"batchsize"           : [10,20]

```
usage: medical_diagnosis_hyperparameter_tuning.py 

optional arguments:
  -h,                   show this help message and exit
  

  --data_dir 
                        Absolute path to the data folder containing
                        "chest_xray" and "chest_xray" folder containing "train" "test" and "val" 
                         and each subfolders contain "Pneumonia" and "NORMAL" folders

```
**Command to run hyperparameter tuning**

```sh
python src/medical_diagnosis_hyperparameter_tuning.py --datadir  ./data/chest_xray
```
By default, model checkpoint will be saved in "model" folder.

> **Note**: Here using --codebatchsize 20 and  --learningRate 0.001 best accuracy has been evaluated ,even that model is compatible for INC conversion

<br>**Convert the model to frozen graph**

Run the conversion module to convert the TensorFlow checkpoint model format to frozen graph format. This frozen graph can be later used for Inferencing, INC and Intel® Distribution of OpenVINO™.
```
usage: python src/model_conversion.py [-h] [--model_dir] [--output_node_names]

optional arguments:
  -h  
                            show this help message and exit
  --model_dir
                            Please provide the Latest Checkpoint path e.g for
                            "./model"...Default path is mentioned

  --output_node_names       Default name is mentioned as "Softmax"
```
**Command to run conversion**

```sh
python src/model_conversion.py --model_dir ./model --output_node_names Softmax
```
> **Note**: We need to make sure intel frozen_graph.pb gets generated using intel model files only 

### 4. Inference

Performed inferencing on the trained model using TensorFlow  2.9.0 with oneDNN

#### Running inference using TensorFlow

```
usage: inference.py [--codebatchsize ] [--modeldir ]

optional arguments:
  -h,                       show this help message and exit

  --codebatchsize           --codebatchsize
                              batchsize used for inference
                        
  --modeldir                --modeldir         
                              provide frozen Model path ".pb" file...users can also
                              use INC INT8 quantized model here

```
**Command to run inference**

```sh
OMP_NUM_THREADS=4 KMP_BLOCKTIME=100 python src/inference.py --codebatchsize 1  --modeldir ./model/updated_model.pb
```
>**Note** : Above inference script can be run in intel environment using different batch sizes<br>

### 5. Quantize trained models using Intel® Neural Compressor

Intel® Neural Compressor is used to quantize the FP32 Model to the INT8 Model. Optimized model is used here for evaluating and timing Analysis.
Intel® Neural Compressor supports many optimization methods. In this case, we used post training quantization with `Default Quantiztion Mode` method to quantize the FP32 model.

>**Note**: We need to make sure intel frozen_graph.pb gets generated using intel model files only .We recommend initiate running hyperparametertuning script with default parameter to get a new model then convert to Frozen graph and using that get the compressed model , if model gets corrupted for any reason below script will not run .

*Step-1: Conversion of FP32 Model to INT8 Model*

```
usage: src/INC/neural_compressor_conversion.py  [--modelpath] ./model/updated_model.pb  [--outpath] ./model/output/compressedmodel.pb [--config]  ./src/INC/deploy.yaml

optional arguments:
  -h                          show this help message and exit

  --modelpath                 --modelpath 
                                Model path trained with TensorFlow ".pb" file
  --outpath                   --outpath 
                                default output quantized model will be save in ".model//output" folder
  --config                    --config 
                                Yaml file for quantizing model, default is "./deploy.yaml"
  
```

**Command to run the neural_compressor_conversion**
> Activate intel Environment before running

```
 python src/INC/neural_compressor_conversion.py  --modelpath  ./model/updated_model.pb  --outpath ./model/output/compressedmodel.pb  --config  ./src/INC/deploy.yaml
```
> Quantized model will be saved by default in `model/output` folder as `compressedmodel.pb`

*Step-2: Inferencing using quantized Model*

```
usage: inference_inc.py [--codebatchsize ] [--modeldir ]

optional arguments:
  -h,                       show this help message and exit

  --codebatchsize           --codebatchsize
                              batchsize used for inference
                        
  --modeldir                --modeldir         
                              provide frozen Model path ".pb" file...users can also
                              use INC INT8 quantized model here

```
**Command to run inference**

```sh
OMP_NUM_THREADS=4 KMP_BLOCKTIME=100 python src/INC/inference_inc.py --codebatchsize 1  --modeldir ./model/updated_model.pb
```
>**Note** : Above inference script can be run in intel environment using different batch sizes<br>
Same script can be used to benchmark INC INT8 Quantized model. For more details please refer to INC quantization section.By using different batchsize one can observe the gain obtained using Intel® oneDNN optimized TensorFlow in intel environment. <br>

Run this script to record multiple trials and the minimum value can be calculated.

*Step-3 : Performance of  quantized Model*

```
usage: src/INC/run_inc_quantization_acc.py  [--datapath]   [--fp32modelpath]  [--config]   [--int8modelpath ]

optional arguments:
  -h,                       show this help message and exit

  --datapath                --datapath
                              need to mention absolute path of data
                        
  ---fp32modelpath          --fp32modelpath         
                              provide frozen Model path ".pb" file...(Absolute path)

  --config                  --config        
                              provide config path...(Absolute path)

  --int8modelpath          --int8modelpath      
                             provide int8 model path ".pb" file...(Absolute path)
                              

```

**Command to run Evalution of INT8 Model**

```sh
python src/INC/run_inc_quantization_acc.py --datapath ./data/chest_xray/val --fp32modelpath ./model/updated_model.pb --config ./src/INC/deploy.yaml --int8modelpath ./model/output/compressedmodel.pb
```

### 6. Quantize trained models using  Intel® Distribution of OpenVINO™

When it comes to the deployment of this model on edge devices, with less computing and memory resources, we further need to explore options for quantizing and compressing the model which brings out the same level of accuracy and efficient utilization of underlying computing resources. Intel® Distribution of OpenVINO™ Toolkit facilitates the optimization of a deep learning model from a framework and deployment using an inference engine on such computing platforms based on Intel hardware accelerators. Below section covers the steps to use this toolkit for the model quantization and measure its performance.

**Intel® Distribution of OpenVINO™ Intermediate Representation (IR) conversion** <br>
Below are the steps to convert TensorFlow frozen graph representation to OpenVINO IR using model optimizer.

*Environment Setup*

Intel® Distribution of OpenVINO™ is installed in OpenVINO environment. Since Intel® Distribution of OpenVINO™ supports Tensorflow<2.6.0.

```sh
conda env create -f env/OpenVINO.yml
```
*Activate OpenVINO environment*
```sh
conda activate OpenVINO
```


Frozen graph model should be generated using `model_conversion.py`, post training from the trained TensorFlow checkpoint model.

**Command to create Intel® Distribution of OpenVINO™ FPIR model**

```sh
mo --input_meta_graph ./model/Medical_Diagnosis_CNN.meta --input_shape="[1,300,300,3]" --mean_values="[127.5,127.5,127.5]" --scale_values="[127.5]" --data_type FP32 --output_dir ./model  --input="Placeholder" --output="Softmax"
```

>>**Note**: The above step will generate `Medical_Diagnosis_CNN.bin` and `Medical_Diagnosis_CNN.xml` as output in `model` which can be used with OpenVINO inference application. Default precision is FP32.

#### Model Quantization

```
python src/OPENVINO/run_openvino_script.py  --datapath ./data/chest_xray/val  --modelpath ./model/Medical_Diagnosis_CNN.xml

optional arguments:
  -h,                     show this help message and exit

  --modelpath,            --modelpath
  
  --datapath              --datapath
                            dataset folder containing "val"
      
```
**Command to run coversion of OpenVINO FPIR model to INT8 model**

```sh
python src/OPENVINO/run_openvino_script.py  --datapath ./data/chest_xray/val  --modelpath ./model/Medical_Diagnosis_CNN.xml
```

> The above step will quantize the model and generate `Medical_Diagnosis_CNN.bin` and `Medical_Diagnosis_CNN.xml` as output in `./model/optimized` which can be used with  Intel® Distribution of OpenVINO throughput and latency benchmarking. post quantization precision is INT8.

#### Benchmarking with  Intel® Distribution of OpenVINO™ Post-Training Optimization Tool

**Running inference using Intel® Distribution of OpenVINO™**<br>Command to perform inference using Intel® Distribution of OpenVINO™. The model needs to be converted to IR format as per the section. 
Post-training Optimization Tool (POT) is designed to accelerate the inference of deep learning models by applying special methods without model retraining or fine-tuning, like post-training quantization.

*Pre-requisites*
-  Intel® Distribution of OpenVINO™ Toolkit
-  Intel® Distribution of OpenVINO IR converted FP32/16 precision model
-  Intel® Distribution of OpenVINO INT8 model converted using FPIR model.

**Performance Benchmarking of full precision (FP32) Model**<br>Use the below command to run the benchmark tool for the FPIR model generated using this codebase for the Pneumonia detection. 

```sh
Latency mode:
benchmark_app -m ./model/Medical_Diagnosis_CNN.xml -api async -niter 120 -nireq 1 -b 1 -nstreams 1 -nthreads 8

Throughput mode:
benchmark_app -m ./model/Medical_Diagnosis_CNN.xml -api async -niter 120 -nireq 8 -b 32 -nstreams 8 -nthreads 8
```

**Performance Benchmarking of INT8 precision Model**<br>Use the below command to run the benchmark tool for the quantized INT8 model. 

```sh
Latency mode:
benchmark_app -m ./model/optimized/Medical_Diagnosis_CNN.xml  -api async -niter 120 -nireq 1 -b 1 -nstreams 1 -nthreads 8

Throughput mode:
benchmark_app -m ./model/optimized/Medical_Diagnosis_CNN.xml  -api async -niter 120 -nireq 8 -b 32 -nstreams 8 -nthreads 8
```

## Performance Observations
 

### Observations
This section covers the inference time comparison between stock TensorFlow and Tensorflow v2.9.0 with oneDNN for this model building. Accuracy of the models both stock and intel during Training and Hyper parameter tuning is above 90%.

#### Inference benchmarking results 

![image](assets/inference.PNG)

<br>**Key Takeaways**<br>

-  Realtime prediction time speedup with Tensorflow 2.9.0 with oneDNN  FP32 Model shows up to 1.46x against stock Tensorflow 2.8.0 FP32 Model
-  Batch prediction time speedup with Tensorflow 2.9.0  with oneDNN FP32 Model shows up to 1.77x against stock Tensorflow 2.8.0 FP32 Model
-  Intel® Neural Compressor quantization offers Realtime prediction time speedup up to  2.14x against stock Tensorflow 2.8.0  FP32 model
-  Intel® Neural Compressor quantization offers batch prediction time speedup  up to 2.42x against stock Tensorflow 2.8.0  FP32  model with batch size up to 648. At larger batch size, stock Tensorflow 2.8.0 was unable to complete without errors.
- No Accuracy drop observed

## Appendix

### **Experiment setup**

| Platform                          | Microsoft Azure: Standard_D8_v5 (Ice Lake)<br>Ubuntu 20.04
| :---                              | :---
| Hardware                          | Azure Standard_D8_V5
| Software                          | Intel® oneAPI AI Analytics Toolkit, TensorFlow
| What you will learn               | Advantage of using TensorFlow (2.9.0 with oneDNN enabled) over the stock TensorFlow (TensorFlow 2.8.0) for CNN model architecture training and inference. Advantage of Intel® Neural Compressor over TensorFlow v2.8.0.

 ### **Running on Windows**
 
 The reference kits commands are linux based, in order to run this on Windows, goto Start and open WSL and follow the same steps as running on a linux machine starting from git clone instructions. If WSL is not installed you can [ install WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

 > **Note** If WSL is installed and not opening, goto Start ---> Turn Windows feature on or off and make sure Windows Subsystem for Linux is checked. Restart the system after enabling it for the changes to reflect.
