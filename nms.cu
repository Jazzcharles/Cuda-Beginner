#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<vector>
#define THREADS 64
/*
#define CUDA_CHECK(condition) \
(\
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
        printf("%s\n", cudaGetErrorString(error));\
    }\
)
*/
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

__device__ inline float devIoU(const float *a, const float *b){
    //a: [5, ] b: [5, ], ymin, xmin, ymax, xmax, score
    float w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]));
    float h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]));
    float intersect = w * h;
    float sa = (a[2] - a[0]) * (a[3] - a[1]);
    float sb = (b[2] - b[0]) * (b[3] - b[1]);
    float _union = sa + sb - intersect;
    float eps = 1e-4;
    return intersect * 1.0 / (_union + eps);
}

__global__ void nms_kernel(float *bbox_dev, unsigned long long *mask_dev, int num_boxes, int col_blocks, float threshold){
    //for each block(c, r) with thread(t, 0), compute the cur_box: r * 64 + t with boxes[c*64 to c*64+63], store to mask_dev
    //bx = c, by = r, t = tx
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    //因为划block时取整，最后一组可能不满, 实际上的row对应block上面的y方向
    const int row_size = min(num_boxes - by * THREADS, THREADS);
    const int col_size = min(num_boxes - bx * THREADS, THREADS);

    __shared__ float sh[THREADS * 5];
    //put [c*64 ~ c*64+63] to share mem, i.e., in parallel: c * 64 + bx, 放入的时候可以并行放
    if(tx < col_size){
        int cols = tx + bx * THREADS;
        #pragma unroll 5
        for(int j = 0; j < 5; j++){
            sh[tx * 5 + j] = bbox_dev[cols * 5 + j];
        }
        __syncthreads();
    }
    
    //compute cur_box at each row: r * 64 + t with shared mem
    if(tx < row_size){
        //compute cur with share mem
        const int cur_box_idx = (by * THREADS) + tx;
        float *cur_box = bbox_dev + cur_box_idx * 5;
        
        int start = 0;
        if(bx == by){
            start = tx + 1;
        }
        
        unsigned long long t = 0;
        for(int i = start; i < col_size; i++){
            if(devIoU(cur_box, sh + tx * 5) >= threshold){
                t |= (1ULL<<tx);
            }
        }   

        const int mask_idx = cur_box_idx * col_blocks + bx;
        mask_dev[mask_idx] = t;
    }
}



float gen_number(int flag){
    float ret = (1.0 * rand()) / RAND_MAX;
    if(flag == 0 or flag == 1)
        ret = max(0.0, ret - 0.5);
    else if(flag == 2 or flag == 3)
        ret = min(1.0, ret + 0.5);
    else
        ret = ret * 1.0;
    return ret;
}

int main(){
    int num_boxes = 12000;
    float threshold = 0.5;
    //(ymin, xmin, ymax, xmax, score)
    int box_dim = 5;
    int col_size = ceil(num_boxes / THREADS);

    size_t size = num_boxes * box_dim * sizeof(float);
    //we use a 64-bit unsigned long long to represent one line, the state of keep or drop
    size_t mask_size = num_boxes * col_size * sizeof(unsigned long long);
    
    float *bbox_host = (float*)malloc(size);
    unsigned long long *mask_host = (unsigned long long*)malloc(mask_size);
    int *keep_host = (int*)malloc(1 * sizeof(int));
    keep_host[0] = 0;

    for(int i = 0; i < num_boxes; i++){
        for(int j = 0; j < box_dim; j++){
            int idx = i * box_dim + j;
            bbox_host[idx] = gen_number(j);

            if(i < 10){
                std::cout<<i<<' '<<j<<' '<<bbox_host[idx]<<std::endl;
            }
        }
    }


    float *bbox_dev = NULL;
    unsigned long long *mask_dev = NULL;
    int *keep_dev = NULL;

    CUDA_CHECK(cudaMalloc((void**)&bbox_dev, size));
    CUDA_CHECK(cudaMalloc((void**)&mask_dev, mask_size));
    CUDA_CHECK(cudaMalloc((void**)&keep_dev, 1 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(bbox_dev, bbox_host, size, cudaMemcpyHostToDevice));
    
    dim3 threadPerBlock(THREADS);
    //(188, 188) each for a 64-bit ULL
    dim3 blockPerGrid(col_size, col_size);

    nms_kernel<<<blockPerGrid, threadPerBlock>>>(bbox_dev, mask_dev, num_boxes, col_size, threshold);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(mask_host, mask_dev, mask_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(keep_host, keep_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost));


    std :: vector<unsigned long long> remv(col_size);
    std :: vector<int> keep_out(num_boxes);
    int num_keeps = 0;
    //keep_out存的是留下来的框们的编号,总共会有num_keeps个

    for(int i = 0; i < num_boxes; i++){
        int u = i / THREADS;
        int v = i % THREADS;
        if((remv[u] & (1ULL << v)) == 0){
            //if not removed, keep it
            keep_out[num_keeps++] = i;
            
            //remove those are similar to the current
            unsigned long long *p = mask_host + i * col_size;
            for(int j = u; j < col_size; j++){
                remv[j] |= p[j];
            }
        }
    }
    std::cout<<num_keeps<<std::endl;


    cudaFree(bbox_dev);
    cudaFree(mask_dev);
    cudaFree(keep_dev);
    free(bbox_host);
    free(mask_host);
    free(keep_host);
    return 0;
}