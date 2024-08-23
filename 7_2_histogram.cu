#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// 2.386ms
template <int INNER_LOOP_SIZE>
__global__ void histgram(int *hist_data, int *bin_data)
{
    // 使用共享内存
    __shared__ int bin[256];

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 初始化共享内存中的bin数组

    {
        bin[tid] = 0;
    }
    __syncthreads();

    // 计算直方图
    for (int i = 0; i < INNER_LOOP_SIZE; i++)
    {
        atomicAdd(&bin[hist_data[gtid * INNER_LOOP_SIZE + i]], 1);
    }
    __syncthreads();

    // 将共享内存中的bin数组累加到全局内存中的bin_data数组

    {
        atomicAdd(&bin_data[tid], bin[tid]);
    }
    // atomicAdd(&bin_data[hist_data[gtid]], 1);
}

bool CheckResult(int *out, int *groudtruth, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (out[i] != groudtruth[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    float milliseconds = 0;
    const int N = 25600000;
    int *hist = (int *)malloc(N * sizeof(int));
    int *bin = (int *)malloc(256 * sizeof(int));
    int *bin_data;
    int *hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));
    ;
    for (int j = 0; j < 256; j++)
    {
        groudtruth[j] = 100000;
    }

    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    const int innerLoopSize = 200;
    int GridSize = std::min((N + 256 - 1) / (256 * innerLoopSize), deviceProp.maxGridSize[0]);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histgram<innerLoopSize><<<Grid, Block>>>(hist_data, bin_data);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(bin, groudtruth, 256);
    if (is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < 256; i++)
        {
            printf("%lf ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram latency = %f ms\n", milliseconds);

    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}