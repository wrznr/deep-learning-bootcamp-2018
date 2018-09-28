# Deep Learning Bootcamp 2018 - Day 4

## Walter de Back: Uncertainty

- Elaborates on optimization possibilities
- Uncertainty in classification: jupyter notebook!
- `training=True`: always assume to be in training mode (i.e. dropout is also active during prediction)

## Jeffrey Kelling: Unsupervised Learning with Autoencoders and Generative Adversarial Networks

- Training using unlabelled data
- More or less a sequence of compression (reduction of features) and decompression (reconstruction of features)
- Loss is computed by comparing reconstruction with original
- Model can be trained on huge amounts of unlabelled data and is afterwards transferred to a supervised setting.
- Words cannot be one-hot-encoded (too many classes!)
- `word2vec` is an option to vectorize words

## Martin Weigert: Regression and Image Restoration

- Impressive slides
- Image to Image task (R^N:R^N)
- Most things stay the same, last activation function changes (i.e. does not squeeze)
- Applications:
  + Style transfer
  + Image restoration
  + Image translation
