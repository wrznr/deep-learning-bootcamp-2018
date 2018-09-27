# Deep Learning Bootcamp 2018 - Day 3

## Sebastian Starke: Regularization

- Prevent overfitting
- Ideally: minimize the expected error over (hidden) probability distribution
- Instead: minimize the error over (representative) training samples
- When?
  + Small datasets
  + Parameter-rich models
- Concept of early stopping
- Introduce validation set!
- Penalize weights by adding a *regularization* term!
- Randomly *dropout* (i.e. ignore) neurons during training!
  + Generalize by despecializing the network
  + Dropout rate is another hyperparameter
- Augment your data (if possible)!
- Normalize the weight updates during training (i.e. *batch normalization*)!
  + »network can focus on learning, not rescaling«
  + Special layer in the network

## Kashif Rasul: Advanced Optimization

- Stochastic gradient decent issues:
  + Local minima
  + Saddle points
  + Condition number (?)
  + Noisy gradients (only a subset of the training data is used for gradient computation)
- *Momentum update*
  + Compute an alternative step direction (using an additional term)
  + »Dampens« the velocity of going in one direction
- Fancy: *Nesterov* momentum
  + Use lookahead
  + Knowing were you gonna be can influence the decent direction
- Different optimizers using the loss are available
  + *Adagrad*
    * As the cache grows, update size decreases
    * Learning may stop early
  + *RMSprop*
    * Introduce *decay rate* (hyperparameter)
    * Keeps cache size constant 
  + *Adam*
    * Momentum plus decay
- Anneal the learning rate!
  + Stepwise decay
  + Exponential decay (costly)
  + 1/t decay
  + Cosine annealing
- Model design
  + `conv-relu-pool` pattern
  + Prefer stack of small conv filters over one large one

## Steffen : Recurrent Neural Networks

- Access information from previous cell/neuron state
- Basically, every timestep is a new network
- 5 basic architectures: 1:1, 1:n, n:1, n:n, n:n (with shift)
- Backpropagation through time for training
- Same-length restriction on input? Not fully clear! https://github.com/keras-team/keras/issues/6776
- Chunked back propagation through time reduces training complexity

## Kashif Rasul: Hyperparameters

- Lots of hyperparameters
- (Semi-)automatic setting possible
- Start with a guess, train, evaluate, refine
- Strategies:
  + Student assistants
    * Completely manual
  + Grid search
    * Search all possible configuration given pre-defined scales/steps
    * `KerasClassifier` and `GridSearchCV` from `sklearn`
  + Random search
    * As the name suggests
    * `KerasClassifier` and `RandomSearchCV` from `sklearn`
  + Bayesian optimization
    * Predict parameters using a model
    * `Hyperas` (`Hyperopt` with `Keras`)

## Walter de Back: Image segmentation

- Partition an image into multiple segments
