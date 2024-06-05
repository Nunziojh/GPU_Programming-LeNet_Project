# LeNet-5 - CUDA implementation
**Report for the GPU Programming course**
### Authors
- **Pietro Mazza** - *s314897*
- **Nunzio Messineo** - *s315067*

### Introduction
The goal of this project is to implement the LeNet-5 convolutional neural network using the CUDA programming model. 
The LeNet-5 architecture was introduced by Yann LeCun in 1998 and is widely used for handwritten digit recognition.
The aim of this work is to develop an effective neural network for digit recognition, optimized by leveraging GPU architecture to parallelize computational calculation.

The network consists of 7 layers, including 3 convolutional layers, 2 subsampling layers, and 2 fully connected layers. The input image size is of 28x28 readapt at 32x32 for our net. The output is a 10-dimensional vector representing the probabilities of the input image belonging to each of the 10 classes. The network was trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. 

### Implementation

Our project is a from scratch project based on the paper "Gradient-Based Learning Applied to Document Recognition" [^1] and the article of Medium "LeNet-5 Complete Architecture" [^2].

The implementation of the LeNet-5 network consists of two main parts: the forward pass and the backward pass. The forward pass is responsible for computing the output of the network given an input image, while the backward pass is responsible for computing the gradients of the network parameters with respect to the loss function. We adopted the gradient descent algorithm to update the weights of the network.

The project is essentially divided into two phases: the first involves implementing a functional neural network, and the second focuses on optimizing and analyzing the code.

#### Part I: Implementation of LeNet-5

We used a bottom up design starting from the composition of the library functions with the main blocks of the network. For each function we tested the correctness of the results and the performance of the implementation. 

Our approach for the testing part started from the unit tests for each function and ones added to the main code, `backward.cu`, implementing the integration tests. All the functions writed in this part are in the library `gpu_functions_base.cu`.

**Dataset**:
For the training and the test of the network we used the MNIST Dataset, a database of handwritten digits has a training set of 60.000 examples, and a test set of 10.000 examples. The managing of the dataset is in `mnist.h`. With this header file each sample is saved in a struct of the type mnist_data.

``` C
typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
	unsigned int label; /* label : 0 to 9 */
} mnist_data;
```

Based on a compilation directive, the MNIST_DATA_TYPE can be chosen with different type of precision.
In our code we used the float type and we convert the data dimensions from 28x28 to 32x32 to precisely follow the network model.

For the forward we strictly followed all the layers of the network as described by the paper [^1]() while for the backward we calculated by hand all the derivatives with respect to the loss for each layer from the output to the input. We then calculate the parameters update based on the learning rate alfa. We found that the best results in the network training using a value for the hyperparamenter alfa of 0.01. The hyperparameters and the constant values used for the code are present in the header file `gpu_functions.h`.

At the end of the implementation part we obtained 90% of accuracy on test dataset of MNIST after 4 epochs of training.

#### Part 2: Optimizing the code

For the optimization part we used all the techiniques learned at GPU Programming course and tested which one perfome better in our code. 

As developing method for an improved version, we take individually all the functions in `gpu_functions_base.cu` and we implemented different techniques for managing the memory usage and the number of threads that works in parallel. As first evaluation criterion we used the timing calculating the differences between the various implementation. For timing we used two tyes of libraries, `<sys/time.h>` in linux for testing on the JetsonNano and `<time.h>` for windows on <!-- INSERIRE MODELLO NVIDIA GPU -->, automatically managed by the define with the define option that change based on the __ linux__ macro.

**Optimization techiniques**
The main optimization techniques we used are:
* *Minimization of code operation*: involves optimizing arithmetic and logic operations to reduce the computational load and improve performance. This optimization strategy is crucial because even though GPUs are designed for parallel execution of a large number of operations, the type and number of operations can still impact the overall efficiency.
    
    Example of convolution in `gpu_functions.cu`:
            
    ``` C
    for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = data[j];
                tmp += filter[j] * val;
            }
            filter += kernel_dim;
            data += in_dim;
        }
    ```
    In this double for loop instead of access at the matrices memory with 
    ```C
     filter[i * kernel_dim + j] 
     data[i * in_dim + j]
    ```
    we move inside each matrix by adding for each iteration of the more external loop their dimension.

* *Using internal registers*: is a powerful optimization technique due to their extremely low latency and high throughput. Registers are the fastest type of memory on a GPU, and utilizing them effectively significantly enhance the performance of the code especially for values ​​reused multiple times in the code or simply as temporary registers.
    
    Example of convolution in `gpu_functions.cu`:

    ``` C
    int index_data = idy * in_dim + idx;
    if(idx < in_dim && idy < in_dim){
        data[index_data] = in[index_data];
    }
    ```

    instead of

    ``` C
    if(idx < in_dim && idy < in_dim){
        data[idy * in_dim + idx] = in[idy * in_dim + idx];
    }
    ```
* *Streams*: a powerful feature that allow for concurrent execution of multiple operations on the GPU, including computation and memory transfers. They enable better utilization of GPU resources by overlapping tasks and reducing idle time. The streams are used in parts of code where the work on some memory areas is independent between the various flows, but which continues to have a sequential operation between the various functions. 
    <!-- CHECK IT -->
    However this technique is not very useful in our case since at the end of each function block you still have to wait for the end of all the streams to be able to go to the next block, slowing down the optimization.

* *Shared memory*: is a key technique used to optimize performance by reducing the time spent on memory access. It allows threads within the same block to efficiently share data and communicate with each other. Shared memory allows to:
    * Reduce Global Memory Access: Access to global memory is relatively slow. By copying data from global memory to shared memory, threads can reuse this data multiple times without incurring the high latency of global memory.
    * Enable Efficient Data Sharing: Threads within the same block can easily share data through shared memory, allowing for efficient implementation of parallel algorithms.
    * Optimize Memory Bandwidth: By coordinating access to shared memory, threads can reduce the number of memory transactions, leading to better utilization of memory bandwidth.
    
    In our code we initialized with the input matrices optimizing the multiple accesses to same values from different threads.

    Example of `gpu_functions.cu`:

    ```C
    extern __shared__ float s_m[];
    float *data = &s_m[kernel_dim * kernel_dim];
    float *filter = &s_m[0];

    int index_data = idy * in_dim + idx;
    if(idx < in_dim && idy < in_dim){
        data[index_data] = in[index_data];
    }

    int index_filter = idy * kernel_dim + idx;
    int offset = kernel_dim * kernel_dim - 1;
    if(idx < kernel_dim && idy < kernel_dim){
        filter[index_filter] = kernel[offset - index_filter];
    }

    __syncthreads();
    ```
    As we can see from this example a single array of shared memory is instantiated (whose dimensions are defined by the host when the call to the kernel function is made). 
    The vector is then divided arbitrarily by maintaining a pointer to the first available address for each of the variables you want to save. The memory is then initialized by accessing the global memory and a syncthreads is done before going to the actual code. 


### Code analysis
<!--
(HOW WE IMPLEMENTED ALL THESE TECHINIQUES AND DIFFERENCES BETWEEN THE BASE CODE)
EXPLAIN ALL THE DIRECTIVES USED, SUMMARIZED BACKWARD.CU
-->


### Results
<!--
(SHOWING THE OUTPUT OF THE PROFILING WITH SOME PLOTS)
-->

#### Accuracy

#### Profiling

### Conclusion

# Bibliography

[^1]
:    Gradient-Based Learning Applied to Document Recognition, 
    *Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
    *Published in*: Proceedings of the IEEE (1998)

[^2]
:   LeNet-5 Complete Architecture,
    [medium article](https://medium.com/codex/lenet-5-complete-architecture-84c6d08215f9)
