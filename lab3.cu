#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 1
#define LOG 1
//-----------Helper functions-------
__host__ int* generate_random_array(int size) {
    srand(time(NULL));
     int* result = (int*)malloc(size * sizeof(int));
      for (int i = 0; i < size; i++) {
        result[i] = rand();
    }
    return result;
}

__host__ int check(int* arr, int number_of_elements) {
    int current_value = arr[0];
    for(int i=1; i<number_of_elements;i++) {
        if(current_value>arr[i]) {
            return 0;
        }
    }
    return 1;
}

__host__ void errorexit(const char *s) {
		printf("\n%s\n",s); 
		exit(EXIT_FAILURE);   
}

__device__ void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ int partition(int* array, int left, int right) {
    int pivot = array[right];
    int i = left - 1;

    for (int j = left; j < right; ++j) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }

    swap(&array[i + 1],&array[right]);
    return i + 1;
}

__global__ void quicksort(int* array, int left, int right) {
	if (left < right) {
        int pivot = partition(array, left, right);
        quicksort<<<1, 1>>>(array, left, pivot - 1);
        quicksort<<<1, 1>>>(array, pivot + 1, right);
    }
}


int main(int argc,char **argv) {
	int size = atoi(argv[1]);
    int* arr = generate_random_array(size);
    int* dSorted = NULL;
    for(int i=0;i<size;i++) {
        if(LOG) printf("%d->", arr[i]);
    }
    printf("\nAfter sorting:\n");

    if (cudaSuccess != cudaMalloc((void **)&dSorted, size * sizeof(int)))
        errorexit("Error allocating memory on the GPU");

    if (cudaSuccess != cudaMemcpy(dSorted, arr, size* sizeof(int), cudaMemcpyHostToDevice))
        errorexit("Error copying results");
    
	quicksort<<<1, 1>>>(dSorted, 0, size-1);

	if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");

  	cudaDeviceSynchronize();

    if (cudaSuccess != cudaMemcpy(arr, dSorted, size * sizeof(int), cudaMemcpyDeviceToHost))
        errorexit("Error copying results");
    
    for(int i=0;i<size;i++) {
        if (LOG) printf("%d->", arr[i]);
    }
    if (DEBUG && !check(arr, size)) {
       printf("Array is not sorted");
    }
    //free memory
    free(arr);
    if (cudaSuccess != cudaFree(dSorted))
        errorexit("Error when deallocating space on the GPU");
}
