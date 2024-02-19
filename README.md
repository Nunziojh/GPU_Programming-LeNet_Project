# LeNet-5 - CUDA implementation
**Report for the GPU Programming course**
### Authors
- **Pietro Mazza** - *s123456*
- **Nunzio Messineo** - *s315067*

### Introduction
The goal of this project is to implement the LeNet-5 convolutional neural network using the CUDA programming model. The LeNet-5 architecture was introduced by Yann LeCun in 1998 and is widely used for handwritten digit recognition. The network consists of 7 layers, including 3 convolutional layers, 2 subsampling layers, and 2 fully connected layers. The input image size is of 28x28 readapt at 32x32 for our net. The output is a 10-dimensional vector representing the probabilities of the input image belonging to each of the 10 classes. The network was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. 

### Implementation

We used a bottom up design starting from the composition of the library functions with the main blocks of the network. For each function we tested the correctness of the results and the performance of the implementation.

The implementation of the LeNet-5 network consists of two main parts: the forward pass and the backward pass. The forward pass is responsible for computing the output of the network given an input image, while the backward pass is responsible for computing the gradients of the network parameters with respect to the loss function. We adopted the gradient descent algorithm to update the weights of the network.