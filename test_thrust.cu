#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
using namespace std;

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a): a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& b) const
    { 
        return a * x + b;
    }
};

void saxpy_fast(float a , thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(a));
}

int main(){
    printf("Ready to test thrust\n");
    int N = 10;
    thrust::host_vector<float> h(N);

    for(int i = 0; i < N; i++)
        h[i] = i;
    
    thrust::device_vector<float> d = h;
    for(int i = 0; i < N / 2; i++)
        d[i] = d[i] * -1;
    
    for(int i = 0; i < N; i++){
        std::cout<<d[i]<<' ';
    }
    std::cout<<endl;

    saxpy_fast(2.0, d, d);

    for(int i = 0; i < N; i++){
        std::cout<<d[i]<<' ';
    }
    std::cout<<endl;


    return 0;
}