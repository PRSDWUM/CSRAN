# CSRAN
Enhancing Cross-Modal Semantic Relations with Graph Attention Network for Image-Text Retrieval
# Introduction
This is the source code of Cross-modal Semantic Relations Attention Network(CSRAN), an approch for Image-Text Retrieval. It is built on top of the SCAN ([Stacked cross attention for image-text matching by Kuang-Huei Lee](https://github.com/kuanghuei/SCAN)) in PyTorch.
# Requirements and Installation
We recommended the following dependencies:
<br>* Python 3.8
<br>* PyTorch 2.0
<br>* NumPy 1.20.0
<!-- <br>* TensorBoard -->
# Download data
Download the dataset files. We use the dataset files created by SCAN([Stacked cross attention for image-text matching by Kuang-Huei Lee](https://github.com/kuanghuei/SCAN)) .
# Training new models
To train Flickr30K and MS-COCO models:
<br> python train.py
