#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;

__global__ void add(const float *a, const float *b, float *c, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void matadd(const float *a, const float *b, float *c, int n, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = i * m + j;
    if(i < n and j < m){
        c[idx] = a[idx] + b[idx];
    }
}

void check_matadd(const float *a, const float *b, const float *c, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            int idx = i * m + j;
            if(a[idx] + b[idx] != c[idx]){
                printf("Not equal !!! \n");
                exit(1);
            }
        }
    }
    printf("Check matadd success !!\n");
}

__global__ void hello(const char *str){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("%s from gpu on thread %d\n", str, idx);
}

__global__ void showgrid(){
    printf("thread: %d, %d %d\nblock Idxs: %d, %d %d\nblock Dims: %d, %d %d\ngrid: %d, %d %d\n\n\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(){
    int n = 1<<14;
    int m = 1<<14;
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
        ha[i] = rand() / (RAND_MAX);
        hb[i] = rand() / (RAND_MAX);
    }

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    //int threadPerBlock = 512;
    //int blockPerGrid = (total + threadPerBlock - 1) / threadPerBlock;
    clock_t st = clock();
    dim3 threadPerBlock(32, 16);
    dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    matadd<<<threadPerBlock, blockPerGrid>>>(da, db, dc, n, m);    
    clock_t ed = clock();
    cout<<"time used: "<<ed-st<<endl;
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

// #define CHECK(call){
//     const cudaError_t err = call
//     if(err != cudaSuccess){
//         printf("Error: %s:%d \n", __FILE__, __LINE__);
//         printf("Code:%d reason: %s\n", err, cudaErrorString(err));
//         exit(1);
//     }
// }
/*
int main(){
    int n = 6;
    dim3 block(3);
    dim3 grid((n + block.x - 1) / block.x);
    printf("on cpu\n");
    printf("block: %d, %d %d\n", block.x, block.y, block.z); 
    printf("grid: %d, %d %d\n", grid.x, grid.y, grid.z);

    showgrid<<<grid, block>>> ();
    cudaDeviceReset();

    return 0;
}
*/

/*
int main(){
    int threadPerBlock = 32;
    int blockPerGrid = (1 + threadPerBlock - 1) / threadPerBlock;
    char *s = new char[10];
    scanf("%s", s);
    size_t size = strlen(s) * sizeof(char);
    char *gpus;
    cudaMalloc((void**)&gpus, size);
    cudaMemcpy(gpus, s, size, cudaMemcpyHostToDevice);
    
    hello<<<blockPerGrid, threadPerBlock>>> (gpus);

    cudaFree(gpus);
    //cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}
*/
/*
int main(){
    int n = 5000000;
    size_t size = n * sizeof(float);
    //cpu中给abc申请内存空间
    float *ha = (float*)malloc(size);
    float *hb = (float*)malloc(size);
    float *hc = (float*)malloc(size);
    float *hd = (float*)malloc(size);
    //init
    for(int i = 0; i < n; i++){
        ha[i] = rand() / (float)(RAND_MAX);
        hb[i] = rand() / (float)(RAND_MAX);
    }
    clock_t st = clock();
    for(int i = 0; i < n; i++){
        hd[i] = ha[i] + hb[i];
    }
    clock_t ed = clock();
    //printf("Time on cpu is %f\n", ed-st);
    cout<<"cpu: "<< ed - st<<endl;

    
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    //给GPU中三个ABC申请显存空间
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    //把数据从cpu内存送到gpu显存中
    cudaMemcpy(d_A, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hb, size, cudaMemcpyHostToDevice);
    st = clock();
    //执行GPU kernel
    int threadPerBlock = 256; //32的倍数
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock; //最少线程块的个数!???
    add<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, n);
    //CHECK(cudaMemcpy(d_C, hc, size, cudaMemcpyDeviceToHost));
    cudaMemcpy(d_C, hc, size, cudaMemcpyDeviceToHost);
    ed = clock();
    cout<<"gpu: "<<ed - st<<endl;

    //printf('Time on gpu is %f\n', ed - st);
    double eps = 1e-5;
    for(int i = 0; i < n; i++){
        if(fabs(ha[i] + hb[i] - hc[i]) < eps){
            fprintf(stderr, "result not same");
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(ha);
    free(hb);
    free(hc);
    //printf("test passed!\n");

    return 0;
}
*/
/*
int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }
    return 0;
}
*/