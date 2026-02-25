### Executing
To compile, install Cuda Toolkit from NVIDIA and run this command in terminal
```
nvcc -o run MNIST.cu && run
```

### Algorithm
I use simple MLP Network with 2 layers:
- Image will be read in binary and flatten to 784-d vector
- W1 (784 x 1024): Feature extractor
- ReLU: activation function of the first layer
- W2 (1024 x 10): Output Layer, calculating raw scores of input image to predict
- Softmax: activation function to output the probabilties

### Technical
- Cuda Programming and Advance C++.
- Implement tiling matrix multiplication to take advantage of GPU ```__shared__``` memory, which reduces number of times communication between ```__global__``` memory and threads.
- Adam Optimizer so the accuracy is around 98%.

### LOVE YOU
Have a nice day~~ 
