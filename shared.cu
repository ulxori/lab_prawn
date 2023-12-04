/*
CUDA - generation and sum of arithmetic progression build of 10240000 elements a1=0 r=1 with shared memory
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
    //allocate memory - size same as number of threads in block
   __shared__ float sresults[1024];  
    int counter;
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    //write in shared memory element value for current thread
    sresults[threadIdx.x]=1.0/pow(2.0, my_index);
    __syncthreads();
    //calculate sum of all elements of thread within same block using shared memory 
   for(counter=512;counter>0;counter/=2) {
      if (threadIdx.x<counter)
        sresults[threadIdx.x]=(sresults[threadIdx.x]+sresults[threadIdx.x+counter]);
      __syncthreads();      
    }

    //first thread in block write results of this block to global memory
    if (threadIdx.x==0) {
      result[blockIdx.x]=sresults[0];
    }
}


int main(int argc,char **argv) {

    long long result;
    int threadsinblock=1024;
    int blocksingrid=10000; 

    int size = threadsinblock*blocksingrid;
    
    //memory allocation on host
    double *hresults=(double*)malloc(blocksingrid*sizeof(double));
    if (!hresults) errorexit("Error allocating memory on the host");  
    
     //devie memory allocation (GPU)
    long *dresults=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&dresults,blocksingrid*sizeof(double)))
      errorexit("Error allocating memory on the GPU");
    
    //call to GPU - kernel execution 
    calculate<<<blocksingrid,threadsinblock>>>(dresults);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
     
      //getting results from GPU to host memory
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,blocksingrid*sizeof(double),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");

    //calculate sum of all elements
    result=0;
    for(int i=0;i<blocksingrid;i++) {
      result = result + hresults[i];
    }

    printf("\nThe final result is %f\n",result);

    //free memory
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");

}
