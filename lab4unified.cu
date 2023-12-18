#include <stdio.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s) {
    printf("\n%s",s);
    exit(EXIT_FAILURE);
}

//elements generation
__global__ void calculate(int numberToTest, int* result) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x + 2;
    if (my_index <= sqrt((float)numberToTest) && numberToTest % my_index == 0) {
      *result = 1;
    }
}

int main(int argc,char **argv) {

    int* result;
    int numberToTest;
    int threadsinblock=1024;
    int blocksingrid=10000;

    //unified memory allocation - available for host and device
    if (cudaSuccess!=cudaMallocManaged(&result,sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    numberToTest = atoi(argv[1]);
    //call to GPU - kernel execution
    calculate<<<blocksingrid,threadsinblock>>>(numberToTest, result);

    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");

    //device synchronization to ensure that data in memory is ready
    cudaDeviceSynchronize();

    if (*result == 1 || numberToTest < 2) {
      printf("Number %d is not prime\n", numberToTest);
    } else {
      printf("Number %d is prime\n", numberToTest);
    }

    //free memory
    if (cudaSuccess!=cudaFree(result))
      errorexit("Error when deallocating space on the GPU");
}
