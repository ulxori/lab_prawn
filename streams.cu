/*
CUDA - generation of array of N elements and calculates even and odd numbers occurence - with streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define MAX 10
__host__ void errorexit(const char *s) {
    printf("\n%s\n",s); 
    exit(EXIT_FAILURE);   
}

__host__ void generate(int *matrix, int matrixSize) {
  srand(time(NULL));
  for(int i=0; i<matrixSize; i++) {
    matrix[i] = rand()%MAX;
  }
}

__global__ void calculation(int *matrix, int* histogram, int matrixSize, int streamChunk, int streamId) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x+streamId*streamChunk;

    if(my_index < matrixSize) {
      if(my_index < matrixSize) {
			  atomicAdd(&histogram[matrix[my_index]], 1);
		  } 
    } 
}

int main(int argc,char **argv) {

  ///define number of streams
  int numberOfStreams = 4;
  
  //define array size and allocate memory on host
  int matrixSize=12;
  int *hMatrix=(int*)malloc(matrixSize*sizeof(int));

  //get number of chunks to operate per stream
  int streamChunk = matrixSize/numberOfStreams;

  printf("Stream chunk is %d \n", streamChunk);
 
  //define kernel size per stream
  int threadsinblock=1000;
  int blocksingrid=1+((streamChunk-1)/threadsinblock); 

  printf("blocksingrid is %d \n", blocksingrid);
  
  //allocate memory for histogram - host
	int *hHistogram=(int*)malloc(MAX * sizeof(int));


  //create streams
  cudaStream_t streams[numberOfStreams];
  for(int i=0;i<numberOfStreams;i++) {
      if (cudaSuccess!=cudaStreamCreate(&streams[i]))
           errorexit("Error creating stream");
    }

  if (cudaSuccess!=cudaMallocHost((void **) &hMatrix, matrixSize*sizeof(int)))
      errorexit("Error allocating memory on the CPU");

  //generate random numbers
  generate(hMatrix, matrixSize);

  if(DEBUG) {
    printf("Generated numbers: \n");
    for(int i=0; i<matrixSize; i++) {
      printf("%d ", hMatrix[i]);
    }
    printf("\n");
  }

  //allocate memory for histogram and matrix- device
	int *dHistogram=NULL;
	int *dMatrix=NULL;

	if (cudaSuccess!=cudaMalloc((void **)&dMatrix,matrixSize*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	if (cudaSuccess!=cudaMalloc((void **)&dHistogram,MAX*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

  //execute operation in each stream - copy chunk of data and run calculations
  for(int i=0; i<numberOfStreams; i++) {
    cudaMemcpyAsync(&dMatrix[streamChunk*i],&hMatrix[streamChunk*i],streamChunk*sizeof(int),cudaMemcpyHostToDevice, streams[i]);      
    calculation<<<blocksingrid, threadsinblock, threadsinblock*sizeof(double), streams[i]>>>(dMatrix, dHistogram, matrixSize, streamChunk, i);
  }

  cudaDeviceSynchronize();

  //copy results from GPU
  if (cudaSuccess!=cudaMemcpy(hHistogram, dHistogram, MAX*sizeof(int),cudaMemcpyDeviceToHost))
		 errorexit("Error copying results");

	for (int i=0; i<MAX; i++) {
		printf("[%d]=%d\n", i, hHistogram[i]);
	}

//Free memory and destroy streams
    for(int i=0;i<numberOfStreams;i++) {
      if (cudaSuccess!=cudaStreamDestroy(*(streams+i)))
         errorexit("Error creating stream");
    }

  free(hHistogram);
  
  if (cudaSuccess!=cudaFreeHost(hMatrix))
     errorexit("Error when deallocating space on the CPU");
  if (cudaSuccess!=cudaFree(dHistogram))
    errorexit("Error when deallocating space on the GPU");
  if (cudaSuccess!=cudaFree(dMatrix))
    errorexit("Error when deallocating space on the GPU");
  
}
