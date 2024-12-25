#ifndef SRC_LAYER_KERNELS_KERNELS_H_
#define SRC_LAYER_KERNELS_KERNELS_H_

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}


class GpuTimer
{
private:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

class cuda_helper
{
private:
    cudaDeviceProp prop;
public:
    void print_device_info();
};


class cuda_manager
{
public:
    void get_device_info();
    void conv_forward(const float* in, float* out, const float* weight, const float* bias, const int n_samples, 
        const int channel_in, const int channel_out, 
        const int height_in, const int width_in, const int kernel_width, const int kernelType);
};

#endif /* SRC_LAYER_KERNELS_KERNELS_H_ */