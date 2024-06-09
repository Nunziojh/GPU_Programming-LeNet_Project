# LeNet-5 - CUDA implementation

### Authors
- **Pietro Mazza** - *s314897*
- **Nunzio Messineo** - *s315067*

### Introduction
The goal of this project is to implement the LeNet-5 convolutional neural network using the CUDA programming model. 
The LeNet-5 architecture was introduced by Yann LeCun in 1998 and is widely used for handwritten digit recognition.
The goal of this work is to develop an effective neural network for digit recognition, optimized by leveraging GPU architecture to parallelize computational calculation.

The network consists of 7 layers, including 3 convolutional layers, 2 subsampling layers, and 2 fully connected layers. The input image size is of 28x28 readapt at 32x32 for our net. The output is a 10-dimensional vector representing the probabilities of the input image belonging to each of the 10 classes. The network was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. 

Our project is a from scratch project based on the paper "Gradient-Based Learning Applied to Document Recognition" [^1] and the article of Medium "LeNet-5 Complete Architecture" [^2].

The project is essentially divided into two phases: the first involves implementing a functional neural network, and the second focuses on optimizing and analyzing the code.

For more information read the report [GPU Programming LeNet Project Report](https://github.com/Nunziojh/GPU_Programming-LeNet_Project/blob/main/report.md)

# Bibliography

[^1] : 
    Gradient-Based Learning Applied to Document Recognition, 
    *Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
    *Published in*: Proceedings of the IEEE (1998)

[^2]
:   LeNet-5 Complete Architecture,
    [medium article](https://medium.com/codex/lenet-5-complete-architecture-84c6d08215f9)
