#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-5


//在原来的情况下，读可以合并，因为可以索引到连续的内存空间（例如行优先），但是写的时候只能跳跃访存写了。
//现在利用shared把一个tile的读进shared进行转置写，就可以高效了
__global__ void mat_transpose(const float *a, float *b, int n, int m){
    const int TIlE_WIDTH = 8;
    __shared__ float temp[TIlE_WIDTH][TIlE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int i = TIlE_WIDTH * bx + tx;
    int j = TIlE_WIDTH * by + ty;
    int idxa = j * n + i;
    int idxb = i * n + j;

    temp[ty][tx] = a[idxa];
    __syncthreads();

    b[idxb] = temp[ty][tx];

    // if(i < n and j < m){
    //     b[idxb] = a[idxa];
    // }
}

void check_mat_transpose(const float *a, const float *b, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            int idxa = i * m + j;
            int idxb = j * m + i;
            //cout<<i<<' '<<j<<' '<<a[idxa]<<' '<<b[idxb]<<endl;
            if(fabs(a[idxa] - b[idxb]) > eps){
                printf("Not equal !!\n");
                exit(1);
            }
        }
    }
    printf("Check matmul success!!!\n");
}


void run_matmul(){
    int n = 1<<7;
    int m = 1<<7;
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
        ha[i] = rand() * 1.0 / (RAND_MAX);
        //hb[i] = rand() * 1.0 / (RAND_MAX);
        //cout<<ha[i]<<' '<<hb[i]<<endl;
    }
    // for(int i = 0; i < total; i++){
    //     cout<<ha[i]<< ' ';
    // }
    // cout<<endl;
    // for(int i = 0; i < total; i++){
    //     cout<<hb[i]<< ' ';
    // }
    // cout<<endl;
    

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    //int threadPerBlock = 512;
    //int blockPerGrid = (total + threadPerBlock - 1) / threadPerBlock;
    //clock_t st = clock();
    dim3 threadPerBlock(8,8);
    dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    mat_transpose<<<blockPerGrid, threadPerBlock>>>(da, db, n, m);    
    //matmul<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);    

    // dim3 threadPerBlock(512);
    // dim3 blockPerGrid((threadPerBlock.x + total - 1) / threadPerBlock.x);
    // matadd_1d<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);
    // dim3 threadPerBlock(512, 1);
    // dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    // matadd_2d<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);


    //clock_t ed = clock();
    //cout<<"time used: "<<ed-st<<endl;
    cudaDeviceSynchronize();
    cudaMemcpy(hb, db, size, cudaMemcpyDeviceToHost);
    //check_matadd(ha, hb, hc, n, m);
    check_mat_transpose(ha, hb, n, m);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);
}


int main(){
    //run_matmul_partition();
    run_matmul();
    return 0;
}