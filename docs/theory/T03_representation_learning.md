## Representation Learning
---

Representation/feature learning is a broader term for techniques that allow an ML system to automatically discover representations needed for further tasks such as feature detection and classification. These tasks require inputs that are mathematically easy to represent and convenient to process and do operations with, however real world data is often the opposite (images, videos, sensor data, etc...). These representations are usually continuous vectors.

### Autoencoders

An autoencoder is a type of neural network architecture that is having three core components: the encoder, the decoder, and the latent-space representation. The encoder compresses the input to a lower latent-space representation via weight matrixes biases and a nonlinear activation function and then the decoder reconstructs it.

### Prototype Learning
Prototype learning is a family of methods where a model represents each class, cluster, or concept using prototypes—vectors in a learned feature space that act as representative examples. A model then makes predictions by comparing new samples to these prototypes. These prototypes provide an abstract representation for many natural categories and concepts.
