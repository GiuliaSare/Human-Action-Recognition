# Human-Action-Recognition

The project was developed by **Sara Nava** and **Giulia Saresini**. Its aim is to develop a neural network capable of classifying seven human actions from the **HMDB51** video dataset.

### Project Overview

The goal of this project is to build a robust neural network model for human action recognition. The model will be trained to classify videos into one of seven predefined human actions: **kick**, **sword**, **kiss**, **hug**, **shake_hands**, **fencing**, **punch**. The HMDB51 dataset contains videos showcasing various human actions, and this project focuses on a subset of these actions.

### Approach

- **Data Preparation**: Preprocess the video data to extract relevant frames and normalize them for training models.
- **Model Development**: Design and train **3D CNN** and **LRCN** architectures suitable for video classification tasks.
- **Evaluation**: Assess the performance of the trained models using metrics such as accuracy and loss.

### Objective

This project represents an initial attempt, acknowledging that the results obtained may not represent the optimal performance achievable. Due to **computational constraints** on our machines, we were limited in experimenting with a smaller number of parameters and simpler models, as more complex models caused kernel crashes. This is why we focused on classifying only 7 out of the 51 action classes available in the HMDB51 dataset. Our goal was not to surpass existing benchmarks but to present a logically structured approach that reflects thoughtful decisions made under these constraints.

### Dataset

The **HMDB51 dataset** includes a diverse range of human actions. Each action category consists of multiple video clips showcasing different instances of the action being performed. For more details, visit the [HMDB51 dataset page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

### References

- [HMDB: A Large Human Motion Database](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Video Classification](https://paperswithcode.com/task/video-classification)
- Karpathy, A., et al. (2015). "Large-scale Video Classification with Convolutional Neural Networks." Available at: [https://arxiv.org/abs/1505.06250](https://arxiv.org/abs/1505.06250)
- [Introduction to Video Classification](https://towardsdatascience.com/introduction-to-video-classification-6c6acbc57356)
- [3D Convolutional Neural Network: A Guide for Engineers](https://www.neuralconcept.com/post/3d-convolutional-neural-network-a-guide-for-engineers)
- [Understanding 1D and 3D Convolution Neural Network (Keras)](https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610)
- [3D Convolutional Neural Network with Kaggle Lung Cancer Detection Competition](https://eitca.org/artificial-intelligence/eitc-ai-dltf-deep-learning-with-tensorflow/3d-convolutional-neural-network-with-kaggle-lung-cancer-detection-competiton/running-the-network-3d-convolutional-neural-network-with-kaggle-lung-cancer-detection-competiton/examination-review-running-the-network-3d-convolutional-neural-network-with-kaggle-lung-cancer-detection-competiton/how-does-a-3d-convolutional-neural-network-differ-from-a-2d-network-in-terms-of-dimensions-and-strides/)
- [Video Classification with a CNN-RNN Architecture](https://www.tensorflow.org/tutorials/video/video_classification)
- [Keras Applications](https://keras.io/api/applications/)
- Simonyan, K., & Zisserman, A. (2014). "Two-Stream Convolutional Networks for Action Recognition in Videos." Available at: [https://arxiv.org/abs/1411.4389?source=post_page](https://arxiv.org/abs/1411.4389?source=post_page)
- [TimeDistributed Layer in Keras](https://keras.io/api/layers/recurrent_layers/time_distributed/)
- [Action Recognition in Videos on HMDB-51](https://paperswithcode.com/sota/action-recognition-in-videos-on-hmdb-51)
