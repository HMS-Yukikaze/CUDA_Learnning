#ifndef _KERNELS_H_
#define _KERNELS_H_

__global__
void
square_array_kernel(float* a, unsigned int numElements);

void
test();

#endif //  _KERNELS_H_