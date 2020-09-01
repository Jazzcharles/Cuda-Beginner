#include<cuda_runtime.h>
#include<stdio.h>
#define CUDA_KERNEL_LOOP(i, n) \
    for(unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define eps 1e-4
const int MOD = 10000;


__global__ void montecarlo(float *dx, float *dy, float *dpi, int N){
    // naive solution: one thread per computation
    
    // int tx = threadIdx.x;
    // int bx = blockIdx.x;
    // int idx = bx * blockDim.x + tx;

    // if(idx < N){
    //     if(dx[idx] * dx[idx] + dy[idx] * dy[idx] < 1.0)
    //         atomicAdd(&dpi[0], 1);
    // }
    
    // another solution: one thread for mutli-computation as N might be larger than total threads
    CUDA_KERNEL_LOOP(i, N){
        if(dx[i] * dx[i] + dy[i] * dy[i] < 1.0)
            atomicAdd(&dpi[0], 1);
    }
        
}
/*
__global__ void montecarlo(float *dx, float *dy, float *dresult, int N){
    // another solution: one thread for mutli-computation as N might be larger than total threads
    CUDA_KERNEL_LOOP(i, N){
        if(dx[i] * dx[i] + dy[i] * dy[i] < 1.0)
            dresult[i] = 1;
        else
            dresult[i] = 0;
    }
        
}

__global__ void accumulate(float *dcount, float *dpi, int N){
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = bx * blockDim.x + tx;
    //reduction_sum, we make tile with size 512
    const int TILE_SIZE = 512;
    __shared__ int sh[TILE_SIZE];
    int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for(int tile = 0; tile < total_tiles; tile++){
        //load data into shared mem
        if(idx + tile * TILE_SIZE < N){
            sh[idx] = dcount[idx + tile * TILE_SIZE];
        }

        __syncthreads();
        //computation with reduction sum
        for(int i = TILE_SIZE/2; i > 0; i /= 2){
            
            __syncthreads();



        }
    }
    for(int i = N / 2; i > 0; i /= 2){
        // 0 + 4, 1 + 5, 2 + 6, 3 + 7        
    }
    
}
*/

float get_random(){
    float ret = (float)(rand())/RAND_MAX;
    return ret;
}


void check(float *x, float *y, float *pi, int N){
    int cnt = 0;
    for(int i = 0; i < N; i++){
        if(x[i] * x[i] + y[i] * y[i] < 1.0)
            cnt++;
    }
    
    float pi_cpu = 4 * cnt * 1.0 / N;
    float pi_gpu = 4 * pi[0] / N;
    printf("%f %f\n", pi_cpu, pi_gpu);
    if(abs(pi_cpu - pi_gpu) < eps){
        printf("OOOOHHHH! U R right !!!\n");
    }
    else{
        printf("...U missed it........\n");
    }
}

int main(){
    //use montecarlo to estimate the pi
    int N = 3e6;

    //default_random_engine e;
    //uniform_real_distribution<float> u(0, 1);
    srand((int)time(0));
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *pi = (float*)malloc(1 * sizeof(float));
    for(int i = 0; i < N; i++){
        x[i] = get_random();
        y[i] = get_random();
        //printf("%f %f\n", x[i], y[i]);
    }
    //check(x, y, pi, N);
    //return 0;
    //sys.exit(0)

    float *dx = NULL;
    float *dy = NULL;
    float *dcount = NULL;
    float *dpi = NULL;
    cudaMalloc((void**)&dx, N * sizeof(float));
    cudaMalloc((void**)&dy, N * sizeof(float));
    cudaMalloc((void**)&dcount, N * sizeof(float));
    cudaMalloc((void**)&dpi, 1 * sizeof(float));
    
    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dcount, 0, N * sizeof(float));

    dim3 threadPerBlock(1024);
    //dim3 blockPerGrid((N + threadPerBlock.x - 1) / threadPerBlock.x);
    dim3 blockPerGrid(1024);
    printf("%d\n", threadPerBlock.x * blockPerGrid.x);
    //call
    montecarlo<<<blockPerGrid, threadPerBlock>>>(dx, dy, dpi, N);
    //montecarlo<<<blockPerGrid, threadPerBlock>>>(dx, dy, dcount, N);

    cudaDeviceSynchronize();
    cudaMemcpy(pi, dpi, 1 * sizeof(float), cudaMemcpyDeviceToHost);  

    check(x, y, pi, N);

    cudaFree(dpi);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dcount);
    free(x);
    free(y);
    free(pi);

    return 0;
}


/*
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
// #include <curand_kernel.h>
//#include "error_checks.h"

__global__ void trial(int seed, bool count_d[], double x_d[], double y_d[]) {
    // curandState state;
    long long id = blockIdx.x * blockDim.x + threadIdx.x;
    // curand_init(seed, id, 0, &state);
    // double x = (double)curand_uniform_double(&state);
    // double y = (double)curand_uniform_double(&state);
    double x = x_d[id], y = y_d[id];
    if(x*x + y*y <= 1) {
        count_d[id] = true;
    }
    else {
        count_d[id] = false;
    }
}

int main(int argc, char* argv[]) {
    int seed = time(NULL);
    long long total = 1e8;  // 默认100万个样本
    int tn = 128;             // 默认128个线程

    if(argc >= 2) {
        total = atoi(argv[1]);  // 从参数中获取样本数
    }
    if(argc >= 3) {
        tn = atoi(argv[2]);     // 从参数中获得线程数
    }
    dim3 threads(tn);
    dim3 blocks((total+tn-1) / tn);
    long long real_total = threads.x * blocks.x;

    bool* count_h = new bool[real_total];
    bool* count_d;
    double* x_h = new double[real_total];
    double* y_h = new double[real_total];
    double* x_d, *y_d;
    for(long long i = 0; i < real_total; i++) {
        x_h[i] = (double)rand() / RAND_MAX;
        y_h[i] = (double)rand() / RAND_MAX;
    }
    cudaMalloc(&count_d, real_total * sizeof(bool));  // 用于保存结果的显存
    cudaMalloc(&x_d, real_total * sizeof(double));    // 随机数数组x
    cudaMalloc(&y_d, real_total * sizeof(double));    // 随机数数组y
    cudaMemcpy(x_d, x_h, real_total * sizeof(double), cudaMemcpyHostToDevice);  // 拷贝随机数数组
    cudaMemcpy(y_d, y_h, real_total * sizeof(double), cudaMemcpyHostToDevice);  // 拷贝随机数数组

    trial<<<blocks, threads>>>(seed, count_d, x_d, y_d);

    cudaMemcpy(count_h, count_d, real_total * sizeof(bool), cudaMemcpyDeviceToHost);

    long long count = 0;
    for(long long i = 0; i < real_total; i++) {
        if(count_h[i]) {
            count++;
        }
    }
    double pi = 4 * (double)count / real_total;

    printf("[+] total = %lld\n", real_total);  // 实际的total可能与参数不同，取决于是否整除
    printf("[+] count = %lld\n", count);
    printf("[+] pi    = %f\n", pi);
    printf("[+] loss  = %e\n", acos(-1) - pi);

    printf("\nBlocks  = %d\n", blocks.x);
    printf("Threads = %d\n", threads.x);

    return 0;
}
*/