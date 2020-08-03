#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-4

// 2d grid 2d block
__global__ void matadd(const float *a, const float *b, float *c, int n, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = j * n + i;
    if(i < n and j < m){
        c[idx] = a[idx] + b[idx];
    }
}

// 1d grid, 1d block
__global__ void matadd_1d(const float *a, const float *b, float *c, int n, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //处理m个数据相加
    if(i < n){
        for(int j = 0; j < m; j++){
            int idx = j * n + i;
            c[idx] = a[idx] + b[idx];
        }
    }
}

//2d grid, 1d block
__global__ void matadd_2d(const float *a, const float *b, float *c, int n, int m){
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    int j =  blockIdx.y;
    if(i < n and j < m){
        int idx = j * n + i;
        c[idx] = a[idx] + b[idx];
    }
}


void check_matadd(const float *a, const float *b, const float *c, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            int idx = i * m + j;
            cout<<a[idx]<<' '<<b[idx]<<' '<<c[idx]<<endl;
            if(a[idx] + b[idx] != c[idx]){
                printf("Not equal !!! \n");
                exit(1);
            }
        }
    }
    printf("Check matadd success !!\n");
}

// __global__ void matmul(const float *a, const float *b, float *c, int n, int m){
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int j = blockDim.y * blockIdx.y + threadIdx.y;
//     int idx = i * m + j;
//     if(i < n and j < m){
//         for(int k = 0; k < n; k++){

//         }
//     }
// }

// void check_matmul(const float *a, const float *b, const float *c, int n, int m){
//     for(int i = 0; i < n; i++){
//         for(int j = 0; j < m; j++){
//             //c[i][j] += a[i][k] * b[k][j];
//             float sum = 0;
//             for(int k = 0; k < n; k++){
//                 sum += a[i][k] * b[k][j];
//             }
//             if(fabs(sum - c[i][j]) > eps){
//                 printf("Not equal !!\n");
//                 exit(1);
//             }
//         }
//     }
//     printf("Check matmul success!!!\n");
// }


int main(){
    int n = 1<<1;
    int m = 1<<1;
    int total = n * m;
    size_t size = (total) * sizeof(float);

    float *ha = (float*)malloc(size);
    float *hb = (float*)malloc(size);
    float *hc = (float*)malloc(size);

    float *da = NULL, *db = NULL, *dc = NULL;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);
    
    for(int i = 0; i < total; i++){
        ha[i] = rand() * 1.0/ (RAND_MAX);
        hb[i] = rand() * 1.0/ (RAND_MAX);
    }

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    //int threadPerBlock = 512;
    //int blockPerGrid = (total + threadPerBlock - 1) / threadPerBlock;
    //clock_t st = clock();
    dim3 threadPerBlock(32,16);
    dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    matadd<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);    

    // dim3 threadPerBlock(512);
    // dim3 blockPerGrid((threadPerBlock.x + total - 1) / threadPerBlock.x);
    // matadd_1d<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);
    
    // fastest
    // dim3 threadPerBlock(512, 1);
    // dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    // matadd_2d<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);


    //clock_t ed = clock();
    //cout<<"time used: "<<ed-st<<endl;
    cudaDeviceSynchronize();
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
    check_matadd(ha, hb, hc, n, m);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);

    return 0;
}