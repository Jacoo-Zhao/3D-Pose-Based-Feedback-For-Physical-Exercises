# 3D-Pose-Based-Feedback-For-Physical-Exercises

### Introduction
Unsupervised self-rehabilitation exercises and physical training can cause serious injuries if performed incorrectly. \ms{We introduce a learning-based framework that identifies the mistakes made by a user and proposes corrective measures for easier and safer individual training. Our framework does not rely on hard-coded, heuristic rules. Instead, it learns them from data, which facilitates its adaptation to specific user needs. To this end, we use a Graph Convolutional Network (GCN) architecture acting on the user's pose sequence to model the relationship between the the body joints trajectories. To evaluate our approach, we introduce a dataset with 3 different physical exercises. Our approach yields 90.9\% mistake identification accuracy and successfully corrects 94.2\% of the mistakes.

![](https://lh5.googleusercontent.com/q0b0XdYewFq0hW48HzMSe0drh7QMqCqSL3H-qGMjt_XkeARnYgAVpki2d5d7y-5myN0=w2400)

Figure1: Gif of Our results. The red poses correspond to the exercises performed incorrectly while the   green poses correspond to our corrections. 


### Framework
![](https://lh6.googleusercontent.com/2PO6dkMio3BkXFS64kEFeesOmkrTeE4Wyjbf5BVIejP_87fwuKDZNHDyvvBPhUR6vyU=w2400)
Figure2: **Our framework**  consists of a classification and a correction branch. They share several graph convolutional layers are then split such that the classification branch identifies the type of mistakes made by the user and the correction branch outputs a corrected pose sequence. The result of the classification branch is fed to the correction branch via a feedback module.

Our framework for providing exercise feedback relies on GCNs which can learn to exploit the relationships between the trajectories of individual joints. The overall model consists of two branches: **the classification branch** which predicts whether the input motion is correct or incorrect, specifying the mistake being made in the latter case, and **the correction branch** that outputs a corrected 3D pose sequence, providing a detailed feedback to the user. We feed the predicted action labels coming from the classification branch to the correction branch, which is called the “feedback module”. It allows us to explicitly provide label information to the correction module, enabling us to further improve the accuracy of the corrected motion.

### Dependencies
-   numpy==1.18.1
-   matplotlib==3.2.1
-   torch==1.4.0
-   torchvision==0.5.0
-   pandas==0.25.0

### Datasets & Data preparation
- You can access all the data used  [coming soon!](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises). 

### Training & Testing
*Coming soon!*

### Reference
This repository holds the code for the following paper:

[3D-Pose-Based-Feedback-For-Physical-Exercises](https://arxiv.org/abs/2208.03257). ACCV, 2022.

If you find our work useful, please cite it as:
```
@inproceedings{zhao2022exercise,
  author = {Zhao, Ziyi and Kiciroglu, Sena and Vinzant, Hugues and Cheng, Yuan and Katircioglu, Isinsu and Salzmann, Mathieu and Fua, Pascal},
  booktitle = {ACCV},
  title = {3D Pose Based Feedback for Physical Exercises},
  year = {2022}
}
```
