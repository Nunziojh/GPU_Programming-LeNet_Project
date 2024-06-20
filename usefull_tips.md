# USEFULL COMPILING DIRECTIVES #

## For running the demo ##

For running the demo you just need to compile with:

> ` nvcc -D USAGE backward.cu gpu_functions.cu `

and then run the ` paint.py ` script.<br>

If you want to use different parameter for the network you need to change the file to use inside the code by replacing the value of the directive ` PARAMETER_FROM_FILE ` with your file.

## For testing the main code ##

In the *main* folder you need to compile ` leNet.cu ` with the following directives:

- One and only one among:
    
    - __TRAIN__ This allows you to train the network on the training part of the MNIST dataset composed of 60'000 images
    - __TEST__ This allows to test the network on the validation part of the MNIST dataset composed of 10'000 images
    - __USAGE__ This allows to use the network on an file called ` input_img.txt ` (with a specific format of 32 x 32 floating point values between 0 and 1) place int the *main* folder

- __PARAMETER_FROM_FILE__ If you want to load existing parameter for your network. The parameter file should have a specific format and this files will be created by the network while training at the end of each epoch.

- __CHECK_PARAMETER_CORRECTNESS__ Is a debugging option that give the possibility to double check that the parameter values are the intended ones.

- __TIME_TEST__ Is a directive to microsecond precision on the execution time.

**BEWARE** that while training the network will report the values of the losses, the predictions, the execution time (in seconds) and the values of the parameter (at the end of each epoch), but compiling with __TEST_TIME__ you lose all this information so may want to hard code the ` epoch_number ` and the ` batch_number ` and use this directives only to execute few iterations.

After specify the directives to use, you also need to link all the source files that the main in going to use, but theme are always the same.

> Ex. To train the network from scratch you nedd to compile with:
>> `  nvcc -D TRAIN leNet.cu support_functions/cpu_functions.cu support_functions/gpu_functions.cu ../pooling/v3/monolithic.cu ../matrix_product/v1/base.cu ../matrix_product/v2/shared.cu ../convolution/v5/monolithic_shared.cu ../convolution/v4/monolithic.cu `

Also you may want to change the ` LEARNING_RATE ` and the ` LOSS_MULTIPLIER ` values in ` leNet.h `

## For testing single function ##

After you move inside the folder of the function you want to test you need to compile the  ` .cu ` file with the following compilation directives:<br>

- __DEBUG_PRINT__ to view the result of the computation in the ` result.txt ` file.
- One and only one of the directive present in the ` .cu ` file that allows to run a specific functions in order to chek its execution time in the ` time.txt ` file.

While compiling you also need to remember to add the relative path to the ` .cu ` file of the version of the function you have decided to test with the directive.

**IMPORTANT**: before compiling change the dimensions of the inputs and outputs in the main ` .h ` file accordingly to the chosen function needs.

> Ex. To test the *convolution_3D_shared* function, you need to be in */convolution*, change the dimension of inputs and output in *convolution_functions.h* to your desire and compile with the following command:<br>
>> ` nvcc -D DEBUG_PRINT -D MONOLITHIC_S_F convolution_test.cu v5/monolithic_shared.cu `