#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(float *C, float *A, float *B, int width, int height){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if(tx >= width || ty >= height)
    return;

  float sum = 0;
  for(int i=0; i<width; ++i){
    sum += A[ty * width + i] * B[i * width + tx];
  }

  C[ty * width + tx] = sum;
}

void constantInit(float *data, int size, float val){
    for (int i = 0; i < size; ++i){
        //data[i] = val;
        data[i] = rand() * 1.0 / (RAND_MAX);
    }
}


void matrixMul(){
  unsigned int width = 2;
  unsigned int height = 2;
  unsigned int size = width * height * sizeof(float);
  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);
  // Initialize host memory
  const float valB = 0.01f;
  constantInit(h_A, width*height, 1.0f);
  constantInit(h_B, width*height, valB);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  //copy host memory to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  //config dims
  dim3 block(16, 16);
  // dim3 grid(width / block.x, height / block.y);
  dim3 grid(1, 1);

  // Excute the kernel
  matrixMulKernel<<<grid, block>>>(d_C, d_A, d_B, width, height);

  // Copy the memory from device to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("Checking computed result for correctness: \n");
  for(int i = 0; i < width * height; i++){
      printf("%f ", h_A[i]);
  }
  printf("\n");

  for(int i = 0; i < width * height; i++){
    printf("%f ", h_B[i]);
  }
  printf("\n");  
  for(int i = 0; i < width * height; i++){
    printf("%f ", h_C[i]);
  }
  printf("\n");


  bool correct = true;
  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6 ; // machine zero

  for (int i = 0; i < width*height; i++){
      double abs_err = fabs(h_C[i] - (width * valB));
      double dot_length = width;
      double abs_val = fabs(h_C[i]);
      double rel_err = abs_err/abs_val/dot_length ;
      if (rel_err > eps)
      {
          printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], (float)(width*height), eps);
          correct = false;
      }
  }
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Free
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}

int main(){
  matrixMul();
}