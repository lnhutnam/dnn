#include "kernels.h"
#define TILE_WIDTH 16

void cuda_helper::print_device_info()
{
	CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("**********GPU Device Properties**********\n");
	printf("Name: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Number of SMs: %d\n", prop.multiProcessorCount);
	printf("Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
	printf("GMEM: %zu bytes\n", prop.totalGlobalMem);
	printf("SMEM per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
	printf("SMEM per Block: %zu bytes\n", prop.sharedMemPerBlock);
	printf("*****************************************\n");
}

// Naive implementation
__global__ void conv_forward_kernel(const float *in, float *out, const float *weight, const float * bias,
                                    const int channel_int, const int channel_out,
                                    const int height_in, const int width_in, const int kernel_width)
{
    const int height_out = height_in - kernel_width + 1;
    const int width_out = width_in - kernel_width + 1;

    int width_grid = (width_out - 1) / TILE_WIDTH + 1;

    int sample_idx = blockIdx.z;
    int map_idx = blockIdx.x;
    int row = (blockIdx.y / width_grid) * blockDim.y  + threadIdx.y;
    int col = (blockIdx.y % width_grid) * blockDim.x + threadIdx.x;

    float acc = 0;

    if (row >= height_out || col >= width_out)
        return;

    int hw_in = height_in * width_in;
    int hw_out = height_out * width_out;

    for (int i = 0; i < channel_int; i++)
    {
        for (int j = 0; j < kernel_width; j++)
        {
            for (int k = 0; k < kernel_width; k++)
            {
                int pixel_row = row + j;
                int pixel_col = col + k;
                acc += in[sample_idx * channel_int * hw_in + i * hw_in +
                            pixel_row * width_in + pixel_col] *
                        weight[map_idx * channel_int * kernel_width * kernel_width +
                            i * kernel_width * kernel_width + j * kernel_width + k];
            }
        }
    }
    out[sample_idx * channel_out * hw_out + map_idx * hw_out + row * width_out + col] = acc;
}

// Using share memory
__global__ void conv_forward_kernel_1(const float *in, float *out, const float *weight, const float * bias, const int B,
                                    const int channel_int, const int channel_out,
                                    const int height_in, const int width_in, const int kernel_width)
{
    int m, h_base, w_base, h, w; 
    int X_tile_width = TILE_WIDTH + kernel_width-1; 
    extern __shared__ float smem[]; 
    float* X_shared = &smem[0]; 
    float* W_shared = &smem[X_tile_width * X_tile_width];

    const int H_out = height_in - kernel_width + 1; 
    const int W_out = width_in - kernel_width + 1; 
    int W_grid = (W_out - 1) / TILE_WIDTH + 1; 

    m = blockIdx.x; 
    h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.y % W_grid) * TILE_WIDTH; 
    
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    h = h_base + ty; 
    w = w_base + tx; 
    int sample_idx = blockIdx.z;
    float acc = 0.; 
    for (int c = 0; c < channel_int; c++)
    {
        if (( ty < kernel_width) && ( tx < kernel_width)) 
        {
            W_shared[ty * kernel_width + tx]= weight[m * channel_int * kernel_width * kernel_width + c * kernel_width * kernel_width + ty * kernel_width + tx];
        }
        __syncthreads(); 

    
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) 
        { 
        for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) 
        {
            if(i < height_in && j < width_in)
            {
                X_shared[(i - h_base) * X_tile_width + (j - w_base)] = in[sample_idx * channel_int * height_in * width_in + width_in * height_in * c + i * width_in + j]; 
            }
            else
            {
                X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0;
            }
        }
        } 
        __syncthreads(); 

        for (int p=0; p<kernel_width; p++) 
        {
        for (int q=0; q<kernel_width; q++) {
            if (((ty+p)<X_tile_width) && ((tx+q)<X_tile_width)) {
                acc += X_shared[(ty+p)*X_tile_width+(tx+q)] * W_shared[p * kernel_width + q];
            }
        }
        }
        __syncthreads(); 
    }

    if( sample_idx < B && m < channel_out && h < H_out && w < W_out) {
        out[sample_idx * channel_out * H_out * W_out + m  * H_out * W_out + h * W_out + w] = acc;
    }
}

__host__ void cuda_manager::conv_forward(const float *in, float *out, const float *weight, const float* bias,
                                         const int n_samples, const int channel_in, const int channel_out,
                                         const int height_in, const int width_in, const int kernel_width, const int kernelType)
{
    int height_out = height_in - kernel_width + 1;
    int width_out = width_in - kernel_width + 1;
    int size_in = n_samples * channel_in * height_in * width_in;
    int size_out = n_samples * channel_out * height_out * width_out;
    int size_weight = channel_out * channel_in * kernel_width * kernel_width;
    int size_bias = channel_out;
    
    float *d_in;
    float *d_out;
    float *d_weight;
    float *d_bias;
    CHECK(cudaMalloc(&d_in, size_in * sizeof(float)));
    CHECK(cudaMalloc(&d_out, size_out * sizeof(float)));
    CHECK(cudaMalloc(&d_weight, size_weight * sizeof(float)));
    CHECK(cudaMalloc(&d_bias, size_bias * sizeof(float)));
    CHECK(cudaMemcpy(d_in, in, size_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, weight, size_weight * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, bias, size_bias * sizeof(float), cudaMemcpyHostToDevice));
    

    // Set grid and block dimensions and launch the kernel
    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int grid = height_grid * width_grid;

    dim3 dimGrid(channel_out, grid, n_samples);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    if(kernelType == 0){
        conv_forward_kernel<<<dimGrid, dimBlock, 0>>>(d_in, d_out, d_weight, d_bias, channel_in, channel_out, height_in, width_in, kernel_width);
    }
    else{
        size_t sharedMemSize = ((TILE_WIDTH + kernel_width - 1) * (TILE_WIDTH + kernel_width - 1) + kernel_width * kernel_width)  * sizeof(float);
        conv_forward_kernel_1<<<dimGrid, dimBlock, sharedMemSize>>>(d_in, d_out, d_weight, d_bias, n_samples, channel_in, channel_out, height_in, width_in, kernel_width);
    }
    CHECK(cudaMemcpyAsync(out, d_out, size_out * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    CHECK(cudaMemcpy(out, d_out, size_out * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_weight));  
}