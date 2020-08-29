#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-4

__global__ void accumulate(float *da, float* ans_device, int N){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bx * blockDim.x + tx;
    //printf("%d\n", idx);
    for(int stride = N / 2; stride > 0; stride >>= 1){
        if(idx < stride){
            da[idx] = da[idx] + da[idx + stride];
        }
        __syncthreads();
    }
    if(idx == 0){
        ans_device[0] = da[idx];
        //printf("ans 0: %f\n", ans_device[0]);
    }
}

float accumulate_cpu(float *da, int size){
    if(size == 1)
        return da[0];
    
    int newsize = size / 2;
    int stride = newsize;

    for(int i = 0; i < newsize; i++){
        da[i] = da[i] + da[i + stride];
    }

    if(size % 2 == 1){
        da[0] = da[0] + da[size - 1];
    }
    else{
        ;
    }
    return accumulate_cpu(da, newsize);
}


void check(float *ha, float *ans_host, int N){
    float sum = 0;
    //cout<<sum<<' '<<ans_host[0]<<endl;
    for(int i = 0; i < N; i++){
        sum += ha[i];
    }  
    
    if(sum == ans_host[0]){
        cout<<"Nice ! Equal !!!"<<endl;
    }
    else{
        cout<<"Bad ! Not Equal !"<<endl;
    }
    
}

int main(){
    int N = 1<<8;
    size_t size = N * sizeof(float);

    float *ha = (float*)malloc(size);
    float *ans_host = (float*)malloc(1*sizeof(float));
    for(int i = 0; i < N; i++)
        ha[i] = 1;
    
    //float ans = accumulate_cpu(ha, N);
    //cout<<ans<<endl;

    
    float *da = NULL;
    float *ans_device = NULL;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&ans_device, 1*sizeof(float));
    
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);

    //dim3 threadPerBlock(N);
    //dim3 blockPerGrid(1);
    dim3 threadPerBlock(32);
    dim3 blockPerGrid((N + threadPerBlock.x - 1) / threadPerBlock.x);
    

    accumulate<<<blockPerGrid, threadPerBlock>>> (da, ans_device, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(ans_host, ans_device, 1*sizeof(float), cudaMemcpyDeviceToHost);

    check(ha, ans_host, N);
    
    free(ans_host);
    free(ha);

    cudaFree(ans_device);
    cudaFree(da);
    
    return 0;
}