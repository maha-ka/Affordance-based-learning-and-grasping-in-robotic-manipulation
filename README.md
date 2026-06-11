# Affordance-based-learning-and-grasping-in-robotic-manipulation

you can visit the site (https://github.com/lqh12345/Jacquard_V2?tab=readme-ov-file#section1) for more information about the dataset and download the full one

## Description of the Conducted Research

This project investigates the problem of planar robotic grasp detection from RGB-D images using the Jacquard dataset. The task consists of predicting a grasp configuration for a parallel-jaw gripper, represented as an oriented rectangle in the image plane. The rectangle encodes: (grasp center position (x, y), orientation angle θ, gripper jaw width, gripper jaw height)

### Problem statement

### Theoretical background 


### Implemented models


### Dataset

Experiments are performed on the Jacquard dataset, which contains: synthetic RGB images, depth maps and grasp annotations in text format.
Preprocessing includes: resize to 256×256 , depth normalization and conversion of annotations into heatmaps
Dataset is split into: 80% training, 20% validation.
