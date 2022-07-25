# Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition (SGN)

## Introduction

Skeleton-based human action recognition has attracted great interest thanks to the easy accessibility of the human skeleton data. Recently, there is a trend of using very deep feedforward neural networks to model the 3D coordinates of joints without considering the computational efficiency. In this work, we propose a simple yet effective semantics-guided neural network (SGN). We explicitly introduce the high level semantics of joints (joint type and frame index) into the network to enhance the feature representation capability. Intuitively, semantic information, i.e., the joint type and the frame index, together with dynamics (i.e., 3D coordinates) reveal the spatial and temporal configuration/structure of human body joints and are very important for action recognition.
In addition, we exploit the relationship of joints hierarchically through two modules, i.e., a joint-level module for modeling the correlations of joints in the same frame and a frame-level module for modeling the dependencies of frames by taking the joints in the same frame as a whole. A strong baseline is proposed to facilitate the study of this field. With an order of magnitude smaller model size than most previous works, SGN achieves the state-of-the-art performance.

