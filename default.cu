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

__global__ void calculation(int *matrix, int* histogram, int matrixSize) {
		int my_index=blockIdx.x*blockDim.x+threadIdx.x;
		if(my_index < matrixSize) {
			atomicAdd(&histogram[matrix[my_index]], 1);
		} 
}

int main(int argc,char **argv) {

	//define array size and allocate memory on host
	int matrixSize=10;
	int *hMatrix=(int*)malloc(matrixSize*sizeof(int));
	
	//generate random numbers
	generate(hMatrix, matrixSize);

	if(DEBUG) {
		printf("Generated numbers: \n");
		for(int i=0; i<matrixSize; i++) {
			printf("%d ", hMatrix[i]);
		}
		printf("\n");
	}

	//allocate memory for histogram - host
	int *hHistogram=(int*)malloc(MAX * sizeof(int));
	
	//allocate memory for histogram and matrix- device
	int *dHistogram=NULL;
	int *dMatrix=NULL;

	if (cudaSuccess!=cudaMalloc((void **)&dMatrix,matrixSize*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	if (cudaSuccess!=cudaMalloc((void **)&dHistogram,MAX*sizeof(int)))
			errorexit("Error allocating memory on the GPU");

	//copy array to device
	if (cudaSuccess!=cudaMemcpy(dMatrix,hMatrix,matrixSize*sizeof(int),cudaMemcpyHostToDevice))
		 errorexit("Error copying input data to device");

	int threadsinblock=1024;
	int blocksingrid=1+((matrixSize-1)/threadsinblock); 

	//run kernel on GPU 
	calculation<<<blocksingrid, threadsinblock>>>(dMatrix, dHistogram, matrixSize);

	//copy results from GPU
	if (cudaSuccess!=cudaMemcpy(hHistogram, dHistogram, MAX*sizeof(int),cudaMemcpyDeviceToHost))
		 errorexit("Error copying results");

	for (int i=0; i<MAX; i++) {
		printf("[%d]=%d\n", i, hHistogram[i]);
	}

	//Free memory
	free(hHistogram);
	free(hMatrix);

	if (cudaSuccess!=cudaFree(dHistogram))
		errorexit("Error when deallocating space on the GPU");
	if (cudaSuccess!=cudaFree(dMatrix))
		errorexit("Error when deallocating space on the GPU");
}
