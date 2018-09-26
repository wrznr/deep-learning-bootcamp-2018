# Deep Learning Bootcamp 2018 - Day 1

## Kashif Rasul: Model validation/selection

- Well-known concepts of
  + held-out data
  + cross validation
- Bias vs. variance (underfit vs. overfit)
- *Validation score* as a measure of model quality: find the *sweet spot* (cf. plot)

## Walter de Back: Basics of deep learning

- 2 subsequent networks:
  + convolutional layers (shrinking the input dimensions)
  + dense layers (fully connected, aka. conventional neural nets)
- Automatic feature extraction is *the* main benefit
- How to train a neural network?
  + Input, weights -> weighted sum (+ bias = z score)-> non-linear activation function -> output
  + Predicted output - expected output = loss
  + Use loss to update network weights
- ToDo: Ask about the bias function! Seems superfluous
- Activation function
  + Non-linearity is necessary
  + Sigmoid: scales input [0,1]
    * Interpretable
    * Convenient derivative
    * Some drawbacks (esp. vanishing gradient, cf. slide)
  + Other activation functions? Other training algorithms?
  + Relu: max(0,x)
    * Piecewise linear
    * Very easy to compute
    * Solves vanishing gradient for x>0
    * Multiple variants
  + https://de.wikipedia.org/wiki/K%C3%BCnstliches_Neuron#Aktivierungsfunktionen
- Also covered [here](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Loss
  + Predicted output - expected output = loss (again)
  + Function of model outputs, averageable
  + Regression:
    * Mean absolute error (= L1 loss): constant gradient, outlier-robust
    * Mean squared error (= L2 loss): gradient increases with loss, outlier-sensitive
    * Huber loss: best of both worlds, additional hyper parameter
    * Log-cosh: best of both worlds
  + Classification:
    * Accuracy: argmax, bad approach due to non-smoothity (steps to large)
    * Softmax: probability distribution over classes, compare to GT distribution using cross entropy
    * Cross entropy: Distance between softmax output and one-hot encoded vector (effectively only taking
      positive classifcations into account)

## Kashif Rasul: Gradient Descent

- Recap
  + Sequence of functional compositions + softmax = ^y
  + Loss computation
  + Optimization by updating the weights in order to minimize loss
- Strategies to change weights
  + Naïve: assign random weights and check loss
  + Fancy: new weights are old weights minus some minor modifications -> following the gradient (convex optimization)
- Computational graphs (?):
  + Derivatives plus chain rule
  + Theoretical description of the gradient computation
- Repetition of morning session in greater mathematical detail
- Weight initialization
  + Impossible: W=0, all neurons do the same
  + Naïve: W=small random numbers, all the weights end up being 0 (mean converges at 0)
  + Naïve: W=large random number in [-1,1], all the weights end up being 1 (gradient converges at 0)
  + Fancy: set variance of weights in order to ensure equality of input and output variance (`* np.sqrt(1.0/n_inputs)`)

## Jeffrey Kelling: Convolutional Neural Networks

- MLP (multi layer perceptron): fully connected NN
  + too many connections
  + overfitting
  + hard to train
- Narrowing the search space: *convolution*
- (Downsampling images: *pooling*)
- *Filter(s)*: multiplication with *small* (3x3xdim, 5x5xdim) sliding window(s) (kernels) to extract a single output activation
- Dimensionality is typically increased through convolution
- Cf. https://www.youtube.com/watch?v=FmpDIaiMIeA&t=121s
- Hierarchical representations
