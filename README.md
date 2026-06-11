# Affordance-based-learning-and-grasping-in-robotic-manipulation



## Abstract

This project focuses on developing and demonstrating an integrated robotic grasping system that combines encoder-decoder based visual perception with PPO reinforcement learning for motion control. First, three models were selected and trained for grasp prediction, the trained models were tested for robustness under gaussian noise, blur and brightness variation. The best model with higher accuracy was chosen for the simulation part. The control system operates in a SAPIEN simulated environment using the Franka Emika Panda robotic arm, which is a custom environment powered by Maniskill platform, RGB-D images were taken for differnet object from an overhead camera on the scene, then a grasp rectangle  prediction process has been done, a pipeline for converting  2D grasp to 6-DoF pose has been created, finally, we set the PPO hyperparameters to be trained for executing motion policies to achieve successful grasps, the positional and orientational errors in the final grasp pose were detected for each object.


### Dataset

Experiments are performed on the Jacquard dataset, which contains: synthetic RGB images, depth maps and grasp annotations in text format.

you can visit the site (https://github.com/lqh12345/Jacquard_V2?tab=readme-ov-file#section1) for more information about the dataset and download the full one
