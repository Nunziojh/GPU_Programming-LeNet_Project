# LeNet-5 - CUDA implementation
**Report for the GPU Programming course**

## Authors
- **Pietro Mazza** - *s314897*
- **Nunzio Messineo** - *s315067*


## Table of Contents

- [Introduction](#introduction)
- [Implementation Overview](#implementation)
    - [Part I - Implementation of LeNet-5](#implementation-of-lenet-5)
        - [MNIST Dataset](#dataset)
    - [Part II - Code Optimization](#optimizing-the-code)
        - [Optimization techiniques](#optimization-techiniques)
- [Code analysis](#code-analysis)
    - [Convolution](#convolution)
    - [Matrix Product](#matrix-product)
    - [Pooling](#pooling)
    - [Activation function](#activation-function)
- [Demo](#demo)
- [Conclusion](#conclusion)


### Introduction
The goal of this project is to implement the LeNet-5 convolutional neural network using the CUDA programming model. 
The LeNet-5 architecture was introduced by Yann LeCun in 1998 and is widely used for handwritten digit recognition.

The aim of this work is to develop an effective neural network for digit recognition, optimized by leveraging GPU architecture to parallelize computational calculation.

<div id="Figure 1" align="center">
    <figure>
     <img src="Report_images\LeNet5_Architecture.jpg" width="497" height="152">
     <figcaption>Figure 1: LeNet-5 Architecture</figcaption>
    </figure>  
</div>

The network consists of 7 layers, including 3 convolutional layers, 2 subsampling layers, and 2 fully connected layers. The input image size is of 28x28 readapt at 32x32 for our net. The output is a 10-dimensional vector representing the probabilities of the input image belonging to each of the 10 classes. The network was trained on the MNIST dataset, which consists of 60.000 training images and 10.000 test images of handwritten digits. 

### Implementation

Our project is a from scratch project based on the paper "Gradient-Based Learning Applied to Document Recognition" [[Article 1](#bibliography)] and the article of Medium "LeNet-5 Complete Architecture" [[Article 2](#bibliography)].

The implementation of the LeNet-5 network consists of two main parts: the forward pass and the backward pass. The forward pass is responsible for computing the output of the network given an input image, while the backward pass is responsible for computing the gradients of the network parameters with respect to the loss function. We adopted the gradient descent algorithm to update the weights of the network.

The project is essentially divided into two phases: the first involves implementing a functional neural network, and the second focuses on optimizing and analyzing the code.

#### Implementation of LeNet-5

We used a bottom up design starting from the composition of the library functions with the main blocks of the network. For each function we tested the correctness of the results and the performance of the implementation. 

Our approach for the testing part started from the unit tests for each function and ones added to the main code, `leNet.cu`, implementing the integration tests. All the functions writed in this part are in the library `leNet.h`.

##### Dataset
For the training and the test of the network we used the MNIST Dataset, a database of handwritten digits has a training set of 60.000 samples, and a test set of 10.000 samples. The managing of the dataset is in `mnist.h`. With this header file each sample is saved in a struct of the type mnist_data.

``` C
typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
	unsigned int label; /* label : 0 to 9 */
} mnist_data;
```

Based on a compilation directive, the MNIST_DATA_TYPE can be chosen with different type of precision.
In our code we used the double type, casted then in a float type, and we convert the data dimensions from 28x28 to 32x32 to precisely follow the network model.

For the forward we strictly followed all the layers of the network as described by the paper [[1](#bibliography)] as in the <A href="#Figure 2">Figure 2</A>.

<div id="Figure 2" align="center">
    <figure>
     <img src="Report_images\LeNet5_architecture_table.jpg" width="503" height="257">
     <figcaption>Figure 2: LeNet-5 Architecture table</figcaption>
    </figure>  
</div>

For the backward pass, we calculated by hand all the derivatives with respect to the loss for each layer, from the output to the input.

In our network, we used specific nomenclature for different parts: C for convolutions, P for pooling, A for activation results, W for weight matrices, F for filters/kernels, and Z for fully connected layer results. Each of these is followed by an increasing number corresponding to their position in the forward pass.

The backward phase started with the calculation of the derivative of Z2 with respect to the Loss (called briefly dZ2).
In the file `leNet.cu`, each code block is preceded by a description of the formula used and the dimensions of all the matrices involved.

At the end of this phase, the parameters are updated based on the learning rate alpha. We found that the best training results for the network were achieved using a value for the hyperparamenter alfa of 0.01. The hyperparameters and the constant values used for the code are present in the header file `leNet.h`.

The results of this first part are pretty good, in fact we obtained 90% of accuracy on test dataset of MNIST after 6 epochs of training.

#### Optimizing the code

For the optimization part we used all the techiniques learned at GPU Programming course and tested which one perfome better in our code. 

As developing method for an improved version, we take individually all the base gpu functions of the first part and we implemented different techniques for managing the memory usage and the number of threads that works in parallel.

As first evaluation criterion we used the timing calculating the differences between the various implementation. For timing we used two tyes of libraries, `<sys/time.h>` in linux for testing on the JetsonNano and `<time.h>` for windows on NVIDIA GeForce MX250, automatically managed by the define with the define option that change based on the __ linux__ macro.

The optimization work was specifically tailored for this code. In particular, working with relatively small inputs, simpler optimizations were preferred over more complex ones. This is because the overhead of some additional operations, done to best parallelize resources and work, in some cases, actually worsened performance. Since the computation is fast, the addition of this overhead is decisive for the code's execution time.

###### **Optimization techiniques**
The main optimization techniques we used are:
* *Minimization of code operation*: involves optimizing arithmetic and logic operations to reduce the computational load and improve performance. This optimization strategy is crucial because even though GPUs are designed for parallel execution of a large number of operations, the type and number of operations can still impact the overall efficiency.
    
    Example of `convolution`:
            
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
    
    Example:

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
    <!-- 
    ESEMPIO IL TAILING NELLA CONVOLUTION PERCHE NON ANDIAMO MAI FUORI SPENDIAMO MEMORIA E TEMPO PER SALVARE DATI IN MEMORIA.
    STREAM NON PORTANO GUADAGNI IN TERMINI DI TEMPO MAGARI IN TEMPO DI RISPORSE. LA SUDDIVISIONE DI ORGANIZZAZIONE DEI THREAD IN STREAM, PER POI EFFETTUARE POCA COMPUTAZIONE E POI ASPETTARE LA RISINCRONIZZAZIONE.
    I FOR NEI KERNEL IN GENERE NON SONO UNA SCELTA SAGGIA, MA NEL NOSTRO CASO E' MEGLIO
    -->
    However this technique is not very useful in our case since the overhead for the creation and the destruction of the streams and their synchronization at the end of each function block, slowing down the performance of the code.

* *Shared memory*: is a key technique used to optimize performance by reducing the time spent on memory access. It allows threads within the same block to efficiently share data and communicate with each other. Shared memory allows to:
    * Reduce Global Memory Access: Access to global memory is relatively slow. By copying data from global memory to shared memory, threads can reuse this data multiple times without incurring the high latency of global memory.
    * Enable Efficient Data Sharing: Threads within the same block can easily share data through shared memory, allowing for efficient implementation of parallel algorithms.
    * Optimize Memory Bandwidth: By coordinating access to shared memory, threads can reduce the number of memory transactions, leading to better utilization of memory bandwidth.
    
    In our code we initialized with the input matrices optimizing the multiple accesses to same values from different threads.

    Example of `convolution`:

    ```C
    extern __shared__ float s_m[];
    float *filter = &s_m[0];
    float *data = &s_m[kernel_dim * kernel_dim];

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
The main files of the project are: `leNet.cu`, `leNet.h`.

**`leNet.cu`**

This file contains the complete implementation of the LeNet-5 architecture. It includes various compilation directives tailored to different usage requirements. All values are parameterized, and depending on the specific directives used, the values of the variables are defined accordingly. The compilation directives present are:

* **TRAIN**: In this case will be taken for performing the training the files from the folder MNIST_Dataset: `train-images.idx3-ubyte` and `train-labels-idx1-ubyte`. Typically the value of epoch_dim and batch_dim are set respectively to 4 and 60.000. If you are doing the training the code will ask you the number of epochs.
* **TEST**: compiling with the TEST directive only the forward is considered. The definition of `PARAMETER_FROM_FILE` is automatically defined because in this case they will be used the network parameters already trained saved in the file whose name is defined in `PARAMETER_FILE`. For the test part will be taken from the folder `MNIST_Dataset` the files: `t10k-images.idx3-ubyte` and `t10k-labels-idx1-ubyte`. The value of *epoch_dim* and *batch_dim* are set respectively to 1 and 10.000.
* **TIME_TEST**: this directive was created to address the need for testing the execution time of the forward and backward passes. Specifically, the execution time will be measured and recorded in a file for *batch_dim* iterations, which is set generally to 1000. Subsequently, the average of these values will be calculated to provide a more reliable method for comparing different versions.
* **USAGE**: this directive enables the compilation in a way that allows the use of the file `paint.py` to manually input a number and send it to the network. This directive permits using only the forward pass and utilizes the `PARAMETER_FROM_FILE` defined in `PARAMETER_FILE`. Unlike the *TEST* directive, it performs the prediction without having the correct label for comparison, and it sets *batch_dim* to 1.
* **CHECK_PARAMETER_CORRECTNESS**: is a compilation directive used for debug the correctness of parameters taking from file or randomly generated.
* **PARAMETER_FILE**: the parameter file from which import the values can be also decided in the compilation fase with the compilation directive -D PARAMETER_FILE.

**`leNet.h`**

In leNet.h, there are the inclusions of our libraries, for device `gpu_functions.cu` and host `cpu_functions.cu`, the inclusions for the main libraries used and the definition of some constant values and some compilation directives.

In *gpu_functions.cu*, the entire library of GPU functions created for the network is present. The prototypes of these functions are included in the corresponding header file, along with constants used in the code.
Optimization of library functions has gone through multiple versions, we will analyze each function below.

#### Convolution

Convolution is one of the most used functions within our code. In order to optimize the network we needed to have a convolution that was as efficient as possible. Based on the optimization techniques used we obtained multiple versions of this function, and they are all present in the *convolution folder* with also a test file for running them as unit.

*   *v1* : Base version of convolution. 
    
    Allows calculating the convolution between two two-dimensional square matrices.

    The block size on the x and y axes is equal to the minimum between the spatial dimensions of the output matrix and 32, so as not to exceed the maximum of 1024 threads per block. The grid size is such that it covers the entire output matrix in case it exceeds 32 in both dimensions.

    After calculating the starting index relative to the input matrix considering the padding (we do not consider the stride because in our context we never use it), we iterate with two nested loops for the entire kernel matrix, checking at each step that we are within the input matrix. If the condition is met, we increment a counter register with the product of an input value and a kernel value. At the end, we assign the value to the output matrix with an increment and not through an assignment so that we can use this function for three-dimensional matrices as well.

    For three-dimensional matrices, we iterate the call to this function to calculate all the faces of the output matrix.

    Depending on the depth of the output matrix and the number of required iterations, this function is called the necessary number of times.

    Therefore, before working on a matrix, it is necessary to reset its values.

*   *v2* : Base version of convolution with shared memory.

    We define two buffers by associating them with global memory for easier access later, then we copy the values of both the input and the kernel into shared memory. Below there is the suddivision of the shared memory used and his initialization by accessing at the global memory.

    ``` C
    extern __shared__ float s_m[];
    float *data = &s_m[0];
    float *filter = &s_m[in_dim * in_dim];

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

    The further difference compared to the previous version is the shifted the start index of the input matrix depending on the thread we are in, avoid some calculation on the two for cycles.

*   *v3* : Enhanced version of that with the shared memory.

    The difference is the use of a supplementary register new_in_dim used in the code and also in the initialization to zero of the shared memory.

    ``` C
    int new_in_dim = in_dim + 2 * padding_f;
    if(idx < new_in_dim && idy < new_in_dim){
        data[idy * new_in_dim + idx] = 0;
    }
    ```

*   *v4* : Monolithic version of the 3D convolution.

    Allows calculating the convolution between an input matrix and N kernel matrices, assuming they all have the same depth.
    I(i, i, b, c) * K(k, k, b, n) = O(o, o, n, 1)
    This function is only used during the forward phase of our network.

    The block size on the x, y, and z axes is equal to the minimum between 10 and the spatial dimensions of the output matrix, so as not to exceed 1000 threads per block. Since we are working with square matrices, we have decided to maintain the square block structure at the expense of some unused threads per block. The grid size is such that it covers the entire output matrix in case it exceeds 10 in all three dimensions.

    After shifting the start index of the kernel to use based on the z of the output matrix we are working on, we iterate with three nested loops over the entire kernel matrix. We increment a counter register with the product of an input value and a kernel value. At the end, we assign the value to the output matrix. We ensure not to exceed the dimensions of the input matrix only on the x and y axes because we have the same number of values on the third axis.

    This version has also another function for the case of n out channels, `convolution_forNOutChannels`, and another one for the full convolution, `full_Convolution`.

*   *v5* : Monolithic version of convolution with shared memory.

    The previous version of convolution with the improvement of the shared memory. The last version of this function.

    This version has also another function for the case of n out channels, `convolution_forNOutChannels_shared`.

 Within the network, we limit the use of convolution operations to three types: one for the forward pass and two for the backward pass, one for computing dA and the other for computing dF. We evaluated the performance for each version we tested.

<div id="Figure 3" align="center">
    <figure>
    <img src="Report_images\grafico_tempi_convoluzione.jpg" width="427" height="237">
    <figcaption>Figure 3: Graph average times of the convolution versions for the Forward</figcaption>
    </figure>  
</div>

#### Matrix Product

The development of matrix multiplication requires multiple functions depending on the usage needs. 

The functions created are:
* `matrix_product`: perform a product row by column between the two input matrices.
* `matrix_transpose_product`: perform the product row by column between the transpose of the first input matrix with the second one.
* `matrix_product_transpose`: perform the product row by column between the first input matrix with the transpose of the second one.
* `matrix_dot_product`: perform a dot product between the two input matrices.
* `matrix_scalar_product`: perform a product between all of the elements of the input 2D matrix with the scalar passed as argoument.
* `matrix3D_scalar_product`: perform a product between all of the elements of the input 3D matrix with the scalar passed as argoument.

We managed to obtain an optimized version with improved memory usage through shared memory only for the first three, since in other cases it would only lead to a performance degradation as there is only one write and one read access for the data. All of this function are present in the *matrix_product folder*.

<div id="Figure 4" align="center">
    <figure>
    <img src="Report_images\avg_matrix_product.png" width="320" height="240">
    <figcaption>Figure 4: Graph average times of the matrices product versions</figcaption>
    </figure>  
</div>

#### Pooling

The pooling functions used in this project are the *average pooling* used during the forward phase and the *inverse average pooling* for the backward phase. 
* **Average pooling** is a down-sampling operation used to reduce the spatial dimensions of the input while preserving important information. In this function we calculate the average of the four values inside the sliding pooling window, and then we put that value in the corrispondent cell in the output matrix.
This function has one improved version:
    - *v2*: monolithic version considering 3D input matrices.
* **Inverse average pooling** is an up-sampling operation with the aim of reverse the effect of average pooling by distributing the values from the smaller feature map back into a larger output feature map. In this function each value in the input matrix is multiplied by the proportional value relative to all values within the Pooling region.
The resulting matrix corresponds to the matrix of derivatives of the Loss function with respect to the inputs.
This function has two improved versions: 
    - *v2*: has a vector of internal register that mantain the values inside the pooling region, avoiding multiple read to the same values 
    - *v3*: is a monolitic version of the v2, considering 3D input matrices.

<div id="Figure 5" align="center">
<figure>
<img src="Report_images\avg_pooling.png" width="320" height="240">
<figcaption>Figure 4: Graph average times of the avg pooling and inverse avg pooling versions</figcaption>
</figure>  
</div>



#### Activation function

Following the LeNet-5 architecture we implemented the **tanh**. We implemented only two versions of this function, the base one and the optmized one. The second has as improvement only the usage of a different formulation of the same equation in a way that we can perform less operations.

*base version:*
``` C
float val = in[idy * w + idx];
float p = expf(val);
float m = expf(-val);

in[idy * w + idx] = (p - m) / (p + m);
```
*opt version:*
```C
float val = in[idy * w + idx];
float v = expf(2 * val);

in[idy * w + idx] = (v - 1) / (v + 1);
```
The memory usage efficiency enhancement with shared memory could not make any improvements since for this function only one write and one read access are made.

### Demo

Within the *demo folder*, we developed a Python script, `paint.py` that allows users to draw numbers and send them to the trained network for testing. The images produced by the script closely resemble those in the MNIST dataset, as shown in the figures below. However, we observe a slightly lower accuracy compared to the results obtained from the test dataset.

<table>
  <tr>
    <td>
      <figure>
        <img src="Report_images\draw_from_script.png" width="140" height="140">
        <figcaption>Figure 5: Draw captured with <code>paint.py</code> of the number 5</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="Report_images\mnist_image_sample.png" width="140" height="140">
        <figcaption>Figure 6: Sample image from MNIST Dataset of the number 5</figcaption>
      </figure>
    </td>
  </tr>
</table>

### Conclusion

In the first phase of our project, implementing the LeNet-5 convolutional neural network from scratch, we achieved an accuracy of 90% on the MNIST test dataset after training for 6 epochs. This was accomplished using the TEST directive, which allowed us to validate the network's performance on unseen data.

For the second phase, focused on optimization, we applied various techniques to enhance the efficiency of our CUDA implementation. Our optimized code demonstrated a significant improvement in performance. Specifically, the execution speed of the optimized version was 28 times faster compared to the base implementation. This substantial increase in speed was verified through detailed timing measurements, ensuring a reliable and consistent comparison between different versions of our implementation.

<div id="Figure 5" align="center">
<figure>
<img src="Report_images\leNet_comparison.png" width="500" height="300">
<figcaption>Figure 7: Performace of base and optimized version of our implementation of LeNet-5</figcaption>
</figure>  
</div>

# Bibliography

[^1]
:    Gradient-Based Learning Applied to Document Recognition, 
    *Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
    *Published in*: Proceedings of the IEEE (1998)

[^2]
:   LeNet-5 Complete Architecture,
    [medium article](https://medium.com/codex/lenet-5-complete-architecture-84c6d08215f9)
