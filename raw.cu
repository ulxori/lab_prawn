/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 without shared memory
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

//elements generation
__global__ void calculate(double *result) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    result[my_index]=1.0/pow(2.0, my_index);
}


int main(int argc,char **argv) {

    double result;
    int threadsinblock=1024;
    int blocksingrid=10000;	

    int size = threadsinblock*blocksingrid;
    //memory allocation on host
    double *hresults=(double*)malloc(size*sizeof(double));
    if (!hresults) errorexit("Error allocating memory on the host");	

    double *dresults=NULL;
    //devie memory allocation (GPU)
    if (cudaSuccess!=cudaMalloc((void **)&dresults,size*sizeof(double)))
      errorexit("Error allocating memory on the GPU");

    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(dresults);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //getting results from GPU to host memory
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,size*sizeof(double),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");


    //calculate sum of all elements
    result=0;

    for(int i=0;i<size;i++) {
      result = result + hresults[i];
    }

    printf("\nThe final result is %f\n",result);

    //free memory
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");

}
