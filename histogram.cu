#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-4


__global__ void cal_hist(float *da, int *hist_da, int N, int M){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bx * blockDim.x + tx;
    if(idx < N){
        // add a lock here to make sure this (read, write) operation atomic.
        atomicAdd(&hist_da[(int)da[idx]], 1);
        //hist_da[(int)da[idx]] += 1;
    }
}

void check(float *ha, int *hist_ha, int N, int M){
    int *result = (int*)malloc(M * sizeof(int));
    for(int i = 0; i < M; i++)
        result[i] = 0;

    for(int i = 0; i < N; i++){
        result[(int)ha[i]]++;
    }

    cout<<"Cpu: "<<endl;
    for(int i = 0; i < M; i++){
        cout<<result[i]<<' ';
    }
    cout<<endl;

    cout<<"Gpu: "<<endl;
    for(int i = 0; i < M; i++){
        cout<<hist_ha[i]<<' ';
    }
    cout<<endl;
}

int main(){
    int N = 1 << 10;
    int M = 10;

    //host 
    size_t size = N * sizeof(float);
    float *ha = (float*)malloc(size);
    

    size_t hist_size = M * sizeof(int);
    int *hist_ha = (int*)malloc(hist_size);
    
    for(int i = 0; i < N; i++){
        ha[i] = i % M;
    }
    //device
    float *da = NULL;
    cudaMalloc((void**)&da, size);
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);

    int *hist_da = NULL;
    cudaMalloc((void**)&hist_da, hist_size);

    dim3 threadPerBlock(32);
    dim3 blockPerGrid((N + threadPerBlock.x - 1) / threadPerBlock.x);

    cal_hist<<<blockPerGrid, threadPerBlock>>>(da, hist_da, N, M);

    cudaDeviceSynchronize();
    cudaMemcpy(hist_ha, hist_da, hist_size, cudaMemcpyDeviceToHost);

    check(ha, hist_ha, N, M);

    cudaFree(da);
    cudaFree(hist_da);
    free(ha);
    free(hist_ha);

    return 0;
}