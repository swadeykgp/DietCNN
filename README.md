#DietCNN

## DietCNN is a Python and PyTorch based Deep Neural Network library that provides *multiplication-free* CNN inference. Pre-trained CNN models can be *transformed* or re-trained to make those multiplication-free. It works best for *embedded systems*, that is with smaller networks and datasets. Direct benefits are: *inference energy* reduction and *model size* reduction. 



# Please have a look at  **A-Sample-Code-Walkthrough-MNIST-LeNet-full**  ([code](https://github.com/swadeykgp/DietCNN/blob/main/MNIST_LENET/MNIST-LeNet-All-Experiments.ipynb)) , for an overall idea of how a network is converted to DietCNN, end-to-end. This also covers all the experiments. There is a similar file for each of the dataset-network combination.

## Table of Contents

1. [Setup](#Requirements-and-Installation)
2. [Examples and Result Reproduction](#Examples-and-Result-Reproduction)

## Setup

Step 1: create basic virtual env:

```
sudo apt install python3.8-venv
python3 -m venv dietcnn
source dietcnn/bin/activate
pip install  -r dietcnn_base.txt

```
## Examples and Result Reproduction

### FPGA Results

cd MNIST_LENET

All directories have the same structure

1. go to individual directory e.g. diet_lenet_fpga
2. Start VITIS HLS
3. Create a new project
4. Add source files <conv.cpp>
5. browse top function cnn_forward
6. Add testbench.cpp
7. Finish

8. Run, c simulate, co-synthesize RTL

The same flow applies to CIFAR_VGG and IMGNET_RESNET

Each directory has a notebook <Dataset,Network>-All-Experiments.ipynb to reproduce the experiments


### Experiments with Sigmoid:

cd lenet_model_sig_supp_sec1.1
python lut_utils_sig.py
python  mnist_eval.py 2 0


### Retraining mode

cd retraining

Retraining the LeNet model
python retraining_lenet_fc.py

(set PARALLEL based on your system cores)


### Inference on discretized images:

cd discretized_inference

Create advanced virtual env:
```
sudo apt install python3.8-venv
python3 -m venv dietcnn_adv
source dietcnn_adv/bin/activate
pip install  -r dietcnn_advanced.txt

run:
python imagenet_symencodetest.py
```

Set dataset path, desired #symbol parameters e.g. 2048, 512
Try to encode first, then decode test.

### Citation

If you find our work useful, please consider citing:

```bibtex
@article{dey2023dietcnn,
author = {Swarnava Dey and Pallab Dasgupta and Partha P Chakrabarti},
journal = {ArXiv preprint arXiv:2305.05274},
title = {DietCNN: Multiplication-free Inference for Quantized CNNs},
year = {2023}
}
```

## License

This code is released under MIT license.

## Acknowledgments

Our code was built using several existing GitHub repositories. I will add those soon...


