# LeNet-5 - CUDA implementation
**Report for the GPU Programming course**
### Authors
- **Pietro Mazza** - *s314897*
- **Nunzio Messineo** - *s315067*

### Introduction
The goal of this project is to implement the LeNet-5 convolutional neural network using the CUDA programming model. 
The LeNet-5 architecture was introduced by Yann LeCun in 1998 and is widely used for handwritten digit recognition.
The goal of this work is to develop an effective neural network for digit recognition, optimized by leveraging GPU architecture to parallelize computational calculation.

The network consists of 7 layers, including 3 convolutional layers, 2 subsampling layers, and 2 fully connected layers. The input image size is of 28x28 readapt at 32x32 for our net. The output is a 10-dimensional vector representing the probabilities of the input image belonging to each of the 10 classes. The network was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. 

### Implementation

Our project is a from scratch project based on the paper "Gradient-Based Learning Applied to Document Recognition" [^1] and the article of Medium "LeNet-5 Complete Architecture" [^2].

The project is essentially divided into two phases: the first involves implementing a functional neural network, and the second focuses on optimizing and analyzing the code.

#### Part I: Implementation of LeNet-5

We used a bottom up design starting from the composition of the library functions with the main blocks of the network. For each function we tested the correctness of the results and the performance of the implementation.

The implementation of the LeNet-5 network consists of two main parts: the forward pass and the backward pass. The forward pass is responsible for computing the output of the network given an input image, while the backward pass is responsible for computing the gradients of the network parameters with respect to the loss function. We adopted the gradient descent algorithm to update the weights of the network.

For the forward we strictly followed all the layers of the network as described by the paper [^1] while for the backward we calculated by hand all the derivatives with respect to the loss for each layer from the output to the input. We then calculate the parameters update based on the learning rate (alfa = 0.01).

We obtained 90% of accuracy on test dataset of MNIST after 4 epochs of training.

#### Part 2: Optimizing the code

For the optimization part we used all the techiniques learned at GPU Programming course and tested which one perfome better in our code. 
<!--
STREAMS
SHARED MEMORY
MINIMIZING CODE OPERATIONS USING INTERNAL REGISTERS
(BEFORE - AFTER - TIMING)
-->

### Code analysis
<!--
(HOW WE IMPLEMENTED ALL THESE TECHINIQUES AND DIFFERENCES BETWEEN THE BASE CODE)
-->

### Results
<!--
(SHOWING THE OUTPUT OF THE PROFILING WITH SOME PLOTS)
-->

#### Accuracy

#### Profiling

### Conclusion

# Bibliography

[^1] : 
    Gradient-Based Learning Applied to Document Recognition, 
    *Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
    *Published in*: Proceedings of the IEEE (1998)

[^2]
:   LeNet-5 Complete Architecture,
    [medium article](https://medium.com/codex/lenet-5-complete-architecture-84c6d08215f9)
