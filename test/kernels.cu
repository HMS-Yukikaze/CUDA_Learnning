#include "kernels.cuh"

__global__
void
square_array_kernel(float* a, unsigned int numElements)
{
  // ### implement me ###
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numElements)
    a[i] = a[i] * a[i];
}

void
test()
{
  float*             a_host;                // pointer to array in host memory
  const unsigned int numElements = 100000000;      // number of elements in the array
  size_t             size        = numElements * sizeof(float);
  a_host = new float[size];                 // allocate array on host

  float* a_device;

  cudaMalloc((void**)&a_device, size);
  cudaMemcpy(a_device, a_host, size, cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size  = (numElements + block_size - 1) / block_size;

  square_array_kernel << < grid_size, block_size >> > (a_device, numElements);

  cudaMemcpy(a_host, a_device, size, cudaMemcpyDeviceToHost);
  cudaFree(a_device);

  delete[] a_host;
}