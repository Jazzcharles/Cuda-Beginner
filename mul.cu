#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "cublas_v2.h"

using namespace std;
#define eps 1e-5


__global__ void matmul(const float *a, const float *b, float *c, int n, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    //printf("%d %d %d %d %d %d\n",blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
    int idx = j * n + i;
    if(i < n and j < m){
        //printf("%d %d %d %d %d %d\n", i, j, idx, a[idx], b[idx], c[idx]);
        float sum = 0;
        for(int k = 0; k < n; k++){
            int idxa = j * n + k;
            int idxb = k * n + i;
            sum += a[idxa] * b[idxb];
        }
        c[idx] = sum;
    }
}

__global__ void matmul_traditional(const float *a, const float *b, float *c, int n, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    //printf("%d %d %d %d %d %d\n",blockDim.x,blockDim.y,blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
    int idx = i * n + j;


    int2 i2 = make_int2(1, 2);
    float4 f4 = make_float4(0, 0, 0, 0);
    f4.x = 0.1, f4.y = 0.2, f4.z = 0.3, f4.w = 0.4;
    //printf("%d %d %f %f %f %f\n", i2.x, i2.y, f4.x, f4.y, f4.z, f4.w);

    if(i < n and j < m){
        //printf("%d %d %d %d %d %d\n", i, j, idx, a[idx], b[idx], c[idx]);
        float sum = 0;
        for(int k = 0; k < n; k++){
            int idxa = i * n + k;
            int idxb = k * n + j;
            sum += a[idxa] * b[idxb];
        }
        c[idx] = sum;
    }
}

void check_matmul(const float *a, const float *b, const float *c, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            //c[i][j] += a[i][k] * b[k][j];
            float sum = 0;
            int idx = i * m + j;
            for(int k = 0; k < n; k++){
                //sum = (sum + a[i][k] * b[k][j]);
                int idxa = i * m + k;
                int idxb = k * m + j;
                sum += a[idxa] * b[idxb];
            }
            //cout<<i<<' '<<j<<' '<<sum<<' '<<c[idx]<<endl;
            if(fabs(sum - c[idx]) > eps){
                printf("Not equal !!\n");
                //exit(1);
            }
        }
    }
    printf("Check matmul success!!!\n");
}

//5.67s
void run_matmul(){
    int n = 1<<13;
    int m = 1<<13;
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
        hb[i] = rand() * 1.0 / (RAND_MAX);
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
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    //int threadPerBlock = 512;
    //int blockPerGrid = (total + threadPerBlock - 1) / threadPerBlock;
    //clock_t st = clock();
    dim3 threadPerBlock(8,8);
    dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);
    matmul_traditional<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n, m);    
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
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
    //check_matadd(ha, hb, hc, n, m);
    //check_matmul(ha, hb, hc, n, m);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);
}

__global__ void matmul_partition(const float *a, const float *b, float *c, int n){
    const int TILE_WIDTH = 8;
    __shared__ float na[TILE_WIDTH][TILE_WIDTH];
    __shared__ float nb[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0;


    //每个线程都会执行整个函数，因此每次都是不一样的(ty, tx)位置
    for(int m = 0; m < n / TILE_WIDTH; m++){
        na[ty][tx] = a[row * n + m * TILE_WIDTH + tx];
        nb[ty][tx] = b[(ty + m * TILE_WIDTH) * n + col];
        __syncthreads();
        //整个tile的值都全了才能继续算
        
        #pragma unroll TILE_WIDTH
        for(int k = 0; k < TILE_WIDTH; k++){
            sum += na[ty][k] * nb[k][tx];
        }
        __syncthreads();
        //算完这一个tile才能再往里写
    }
    c[row * n + col] = sum;
}

//3.33s
void run_matmul_partition(){
    int n = 1<<13;
    int m = 1<<13;
    int total = n * m;
    size_t size = total * sizeof(float);
    float *ha = (float*)malloc(size);
    float *hb = (float*)malloc(size);
    float *hc = (float*)malloc(size);

    float *da, *db, *dc;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(8, 8);
    dim3 blockPerGrid((n + threadPerBlock.x - 1) / threadPerBlock.x, (n + threadPerBlock.y - 1) / threadPerBlock.y);

    matmul_partition<<<blockPerGrid, threadPerBlock>>>(da, db, dc, n);
    cudaMemcpy(hc, dc, size, cudaMemcpyHostToDevice);

    //check_matmul(ha, hb, hc, n, n);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);

}

//104ms....
void run_matmul_cublas(){
    int n = 1<<13;
    int m = 1<<13;
    int k = 1<<13;
    int total = n * m;
    size_t size = (total) * sizeof(float);

    cublasStatus_t stat;
    cublasHandle_t handle;

    float *ha = (float*)malloc(size);
    float *hb = (float*)malloc(size);
    float *hc = (float*)malloc(size);

    float *da = NULL, *db = NULL, *dc = NULL;
    cudaMalloc((void**)&da, size);
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);
    
    for(int i = 0; i < total; i++){
        ha[i] = rand() * 1.0 / (RAND_MAX);
        hb[i] = rand() * 1.0 / (RAND_MAX);
    }

    //cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(8,8);
    dim3 blockPerGrid((n+threadPerBlock.x-1)/threadPerBlock.x, (m+threadPerBlock.y-1)/threadPerBlock.y);

    
    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(m, k, sizeof(*ha), ha, m, da, m);
    stat = cublasSetMatrix(k, n, sizeof(*hb), hb, k, db, k);
    stat = cublasSetMatrix(m, n, sizeof(*hc), hc, m, dc, m);
    float alpha = 1.0f, beta = 1.0f;
    stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da,
        m, db, k, &beta, dc, m);
    stat = cublasGetMatrix(m, n,sizeof(*hc), dc, m, hc, m);

    cudaDeviceSynchronize();
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    //check_matmul(ha, hb, hc, n, m);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cublasDestroy(handle);

    free(ha);
    free(hb);
    free(hc);
}

//just a demo code, need modification to run.
//ref: dlsys.cs.washington.edu/pdf/lecture6.pdf
void run_matmul_partition_cpu(){
    int n = 10;
    int t1 = 5;
    int t2 = 5;
    int t3 = 5;
    
    int a[n][n];
    int b[n][n];
    int c[n][n];

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            a[i] = rand() * 1.0 / (RAND_MAX);
            b[i] = rand() * 1.0 / (RAND_MAX);
            c[i] = 0;
        }
    }

    //naive mm
    //memory --> register load cost: n^3 + n^3
    //register load cost: 3
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = 0;
            for(int k = 0; k < n; k++){
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    //output_tiled_mm
    //memory --> register load cost: n^3 / t1 + n^3 / t2
    //register load cost: t1*t2 + t1*t3 + t2*t3
    int A[n / t1][n / t3][t1][t3];
    int B[n / t3][n / t2][t3][t2];
    int C[n / t1][n / t2][t1][t2];

    for(int i = 0; i < n / t1; i++){
        for(int j = 0; j < n / t2; j++){
            register int c[t1][t2] = 0;
            for(int k = 0; k < n / t3; k++){
                register int a[t1][t3] = A[i][k];
                register int b[t3][t2] = B[k][j];
                c += dot_product(a, b);
            }
            C[i][j] = c;
        }
    }

    //cacheline aware tiling
    //memory --> register load cost: n^2/t1 + n^3/t2
    //register: t1*n + t2 * n
    int A[n / t1][t1][n];
    int B[n / t2][t2][n];
    int C[n / t1][t1][n / t2][t2];
    for(int i = 0; i < n / t1; i++){
        l1cache a[t1][n] = A[i];
        for(int j = 0; j < n / t2; j++){
            l1cache b[t2][n] = B[j];
            c[i][j] = dot(a, b);
        }
    }

    //naive combination
    //(1) cacheline --> (2) output tiling
    //L1cache tile with (t1, t2)
    //output tile with (b1, b2), s.t. t1 % b1 == 0, t2 % b2 == 0
    for(int i = 0; i < n / t1; i++){
        l1cache a[t1][n] = A[i];
        for(int j = 0; j < n / t2; j++){
            l1cache b[t2][n] = B[j];
            // replace c[i][j] = dot(a, b); with output tiling
            l1cache c[t1][t2] = 0;
            for(int x = 0; x < t1 / b1; x++){
                for(int y = 0; y < t2 / b2; y++){
                    register int newc[b1][b2] = 0;
                    for(int z = 0; z < n / b3; z++){
                        register int newa[b1][b3] = a[x][z];
                        register int newb[b2][b3] = b[y][z];
                        newc += dot_product(newa, newb); 
                    }
                    c[x][y] = newc;
                }
            }
            C[i][j] = c[t1][t2];
        }
    }

    //same way
    //dram: n^2 + n^3/b1
    //l1cache: n^3/b1 + n^3/b2
    int A[n / t1][t1 / b1][n][b1];
    int B[n / t2][t2 / b2][n][b2];
    for(int i = 0; i < n / t1; i++){
        l1cache a[t1 / b1][n][b1] = A[i];
        for(int j = 0; j < n / t2; j++){
            l1cache b[t2 / b2][n][b2] = B[j];

            licache c[t1][t2] = 0;
            for(int x = 0; x < t1 / b1; x++){
                for(int y = 0; y < t2 / b2; y++){
                    register int newc[b1][b2] = 0;
                    for(int k = 0; k < n; k++){
                        register int newa[b1] = a[x][k];
                        register int newb[b2] = b[y][k];
                        newc += dot_product(newa, newb);
                    }
                    c[x][y] = newc;
                }
            }

            C[i][j] = c[t1][t2];
        }
    }

}

int main(){
    run_matmul_partition();
    //run_matmul();
    //run_matmul_cublas();
    return 0;
}