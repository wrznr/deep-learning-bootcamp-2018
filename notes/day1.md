# Deep Learning Bootcamp 2018 - Day 1

## Uwe Schmidt: Introduction

### Artificial intelligence

- Historical placement
  + *narrow* vs. *general* vs. *super*
  + Term created in 1956
  + *Golden years* until 1974
  + From 1980: focus on knowledge (i.e. expert systems)
  + During the nineties, emerging of subfields
  + Rising popularity in the 21th century
    * Machine learning
    * Big data
    * Deep learning
- AI risks
  + ...

### Machine learning

- Models explain data to allow for subsequent prediction
- Evaluate parameter setting given a *cost* function
- Types of learning
  + Unsupervised learning
  + Supervised learning
  + Reinforcement learning
- Focus on supervised learning
  + Model
  + (GT) data
  + Loss function (GT vs. prediction)
  + Cost function (Loss for all data points)
  + Learning: find the model (parameter setting) with minimal costs
- Continuous output vs. discrete output (regression vs. classification)
- Model (class) selection
  + Acquisition of data
  + Cleaning of data
  + Segmentation?
  + Tracking?
- Comparison of learning to other models: decision trees, SVMs, nearest neighbors
- [Check](playground.tensorflow.org)
- Problem: It is often hard to understand why Neural Networks perform as they do.
- Feature engineering is (theoretical) superfluous.

## Jeffrey Kelling, Sebastian Starke: Classifying Cats and Dogs

- Example classification task: cats and dogs
- Image classification as a standard DL scenario
- Pixel intensities x color channels (3D vectors)
- Benchmark datasets
  + MNIST
  + CIFAR10
  + CIFAR100
  + ImageNet
- Exercise
  + Jupyter Notebooks!
  + Check loss function for training and test
  + Model parameters are reduced using convolutional layers
  + Deterministic training is hard to achieve
  + Parallelization (GPU training) makes training inherently non-deterministic
- Transfer learning
  + Use parts of other (pre-trained) networks
  + Tailor them to your problem
