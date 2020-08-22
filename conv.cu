#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-4

//每个thread负责output的一个pixel, 暂未考虑paddding, 即out_size = (in_size - k + 1)
__global__ void convolution(float *img, float *kernel, float* result, int n, int m, int kw, int kh, int out_n, int out_m)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int idx = y * out_m + x;
    //printf("%d %d %d %d %d %d\n", bx, by, tx, ty, x, y);
    if(idx < out_n * out_m){
        float ret = 0;
        for(int i = 0; i < kw; i++){
            for(int j = 0; j < kh; j++){
                ret += img[(y + j) * m + (x + i)] * kernel[i * kh + j];
            }
        }
        //printf("%d %d %d %f\n", x, y, idx, ret);
        //__syncthreads();
        result[idx] = ret;
    }
}

bool check(float *img, float *kernel, float *result, int n, int m, int kw, int kh, int out_n, int out_m){
    for(int i = 0; i < out_n; i++){
        for(int j = 0; j < out_m; j++){
            float cur = 0.0;
            for(int p = 0; p < kw; p++){
                for(int q = 0; q < kh; q++){
                    cur += img[(i + p) * m + (j + q)] * kernel[p * kh + q];
                }
            }
            //printf("%f %f\n", cur, result[i * out_m + j]);
            //printf("%f\n", cur);
            if(abs(cur - result[i * out_m + j]) > eps){
                cout<<cur<<' '<<result[i * out_m + j]<<endl;
                cout<<"Not Equal !!!"<<endl;
                exit(0);
            }
            //cout<<endl;
        }
    }
    cout<<"Nice !!! Equal!!"<<endl;
    return true;
}

int main(){
    int n = 512;
    int m = 512;
    int kh = 3;
    int kw = 3;
    int out_n = (n - kw + 1);
    int out_m = (m - kh + 1);
    size_t sizer = sizeof(float);
    float *kernel = NULL;
    kernel = (float*)malloc(kw * kh * sizer);

    for(int i = 0; i < kw; i++){
        for(int j = 0; j < kh; j++){
            kernel[i * kh + j] = 1;
        }
    }

    float *img = NULL;
    img = (float*)malloc(n * m * sizer);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            img[i * m + j] = (i + j) % 256;
            //cout<<img[i * m + j]<<' ';
        }
        //cout<<endl;
    }
    
    float *result = (float*)malloc(out_m * out_n * sizer);

    float *img_d = NULL;
    float *kernel_d = NULL;
    float *result_d = NULL;
    cudaMalloc((void**)&kernel_d, kh * kw * sizer);
    cudaMalloc((void**)&img_d, n * m * sizer);
    cudaMalloc((void**)&result_d, out_m * out_n * sizer);

    cudaMemcpy(img_d, img, n * m * sizer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel, kh * kw * sizer, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock(2, 2);
    dim3 BlockPerGrid((out_n + threadPerBlock.x - 1) / threadPerBlock.x, (out_m + threadPerBlock.y - 1)/threadPerBlock.y);

    convolution<<<BlockPerGrid, threadPerBlock>>>(img_d, kernel_d, result_d, n, m,  kw, kh, out_n, out_m);

    cudaDeviceSynchronize();
    cudaMemcpy(result, result_d, out_n * out_m * sizer, cudaMemcpyDeviceToHost);
    
    // for(int i = 0; i < out_n; i++){
    //     for(int j = 0; j < out_m; j++){
    //         cout<<result[i * out_m + j]<<' ';
    //     }
    //     cout<<endl;
    // }

    check(img, kernel, result, n, m, kw, kh, out_m, out_n);

    free(img);
    free(kernel);
    free(result);

    cudaFree(img_d);
    cudaFree(kernel_d);
    cudaFree(result_d);

    return 0;

}