Cuda C for Beginners

Currently supported ops:
1. matadd 
2. matmul
3. transpose
4. im2col
5. naive convolution
6. accumulation (reduced sum)

Currently supported features:
1. Make tile with shared memory


How to use:
1. nvcc mul.cu -o mul
2. ./mul or nvprof ./mul

We will add makefiles in the future
