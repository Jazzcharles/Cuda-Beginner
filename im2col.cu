#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;
#define eps 1e-4

__global__ void im2col(float ***img, float **img_flat, int c1, int n, int m, int kw, int kh, int out_n, int out_m){
    //each thread process a [c1 * kw * kh] flatten op
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = bx * blockDim.x + tx;

    if(idx < out_n * out_m){
        int i = idx / out_m;
        int j = idx % out_m;
        int col = c1 * kw * kh;
        //printf("%d %d\n", i, j);
        
        for(int c = 0; c < c1; c++){
            for(int p = 0; p < kw; p++){
                for(int q = 0; q < kh; q++){
                    int x = idx;            
                    int y = c * kw * kh + p * kh + q;
                    int newidx = x * col + y;
                    
                    int oldx = c;
                    int oldy = p + i;
                    int oldz = q + j;
                    int oldidx = oldx * n * m + oldy * m + oldz;
                    *((float*)img_flat + newidx) = *((float*)img + oldidx);
                    //printf("%d %d %f %f\n", newidx, oldidx, *((float*)img_flat + newidx), *((float*)img + oldidx));
                }
            }
        }
    }
}


void im2col_cpu(float ***img, float **img_flat, int c1, int n, int m, int kw, int kh, int out_n, int out_m){
    //img: [C1, N, M]
    //kernel: [C2, C1, kw, kh]
    //return: img_flat[out_n*out_m, C1*kw*kh]

    int row = out_n * out_m;
    int col = c1 * kw * kh;

    for(int i = 0; i < out_n; i++){
        for(int j = 0; j < out_m; j++){
            for(int c = 0; c < c1; c++){
                for(int p = 0; p < kw; p++){
                    for(int q = 0; q < kh; q++){
                        //printf("%d %d %d %d %d\n", i * out_m + j, c * kw + (p * kh + q), c, p+i, q+j);
                        //img_flat[i * out_m + j][c * kw * kh + (p * kh + q)] = img[c][p+i][q+j];
                        int x = i * out_m + j; 
                        int y = c * kw * kh + (p * kh + q);
                        int idx = x * col + y;

                        int oldx = c;
                        int oldy = p + i;
                        int oldz = q + j;
                        int oldidx = oldx * n * m + (oldy * m + oldz);
                        *((float*)img_flat + idx) = *((float*)img + oldidx);
                    }
                }
            }
        }
    }
}

bool check(float **img_flat, float **img_flat_cpu, int c1, int n, int m, int kw, int kh, int out_n, int out_m){
    int row = out_n * out_m;
    int col = c1 * kw * kh;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            int idx = i * col + j;
            float x = *((float*)img_flat + idx);
            float y = *((float*)img_flat_cpu + idx);
            //cout<<x<<' '<<y<<endl;
            if(abs(x - y) > eps){
                cout<<"Not Equal!!!"<<endl;
                exit(0);
            }
        }
    }
    cout<<"Oh Nice Equal!!!"<<endl;
    return true;
}

int main(){
    //feature map size[C1, N, M]
    int in_channel = 32;
    int n = 64;
    int m = 64;
    size_t size = sizeof(float);

    //kernel size
    int kw = 3;
    int kh = 3;

    //output featuremap size[out_n * out_m, C1 * kw * kh]
    int out_n = n - kw + 1;
    int out_m = m - kh + 1;
    int row = out_n * out_m;
    int col = in_channel * kw * kh;

    // sizes
    int img_size = in_channel * n * m * size;
    int imgflat_size = out_n * out_m * in_channel * kw * kh * size;


    float ***img = (float***)malloc(img_size);
    float **img_flat = (float**)malloc(imgflat_size);
    float **img_flat_cpu = (float**)malloc(imgflat_size);

    float ***img_d = NULL;
    float **img_flat_d = NULL;

    cudaMalloc((void**)&img_d, img_size);
    cudaMalloc((void**)&img_flat_d, imgflat_size);

    for(int c = 0; c < in_channel; c++){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                int idx = c * n * m + i * m + j;
                *((float*)img + idx) = (i + j) % 255;
            }
        }
    }
    
    im2col_cpu(img, img_flat_cpu, in_channel, n, m, kw, kh, out_n, out_m);
    // for(int i = 0; i < row; i++){
    //     for(int j = 0; j < col; j++){
    //         int idx = i * col + j;
    //         cout<<*((float*)img_flat_cpu + idx)<<' ' ;
    //     }
    //     cout<<endl;
    // }

    cudaMemcpy(img_d, img, img_size, cudaMemcpyHostToDevice);
 
    
    dim3 threadPerBlock(32);
    dim3 BlockPerGrid((out_n * out_m + threadPerBlock.x - 1) / threadPerBlock.x);

    im2col<<<BlockPerGrid, threadPerBlock>>>(img_d, img_flat_d, in_channel, n, m, kw, kh, out_n, out_m);

    cudaDeviceSynchronize();
    cudaMemcpy(img_flat, img_flat_d, imgflat_size, cudaMemcpyDeviceToHost);

    // cout<<endl;
    // for(int i = 0; i < row; i++){
    //     for(int j = 0; j < col; j++){
    //         int idx = i * col + j;
    //         cout<<*((float*)img_flat + idx)<<' ' ;
    //     }
    //     cout<<endl;
    // }





    check(img_flat, img_flat_cpu, in_channel, n, m, kw, kh, out_n, out_m);
    
    free(img);
    free(img_flat);
    free(img_flat_cpu);

    cudaFree(img_d);
    cudaFree(img_flat_d);
    //float img[2][2][2]={1, 2, 3, 4, 5, 6, 7, 8};
    //float result[4][2] = {0};
    //float kernel[1][1][1];
    //im2col_cpu((float***)img, (float**)result, 2, 2, 2, 1, 1, 2, 2);
    // for(int i = 0; i < 4; i++){
    //     for(int j = 0; j < 2; j++){
    //         cout<<result[i][j]<<' ';
    //     }
    //     cout<<endl;
    // }


    return 0;
}