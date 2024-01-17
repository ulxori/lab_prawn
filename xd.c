#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define DEBUG 1
#define LOG 1

//-----------Helper functions-------
int* generate_random_array(int size) {
    srand(time(NULL));
    int* result = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        result[i] = rand() % 10000;
    }
    return result;
}

int check(int* arr, int number_of_elements) {
    int current_value = arr[0];
    for (int i = 1; i < number_of_elements; i++) {
        if (current_value > arr[i]) {
            return 0;
        }
    }
    return 1;
}


void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

int partition(int* array, int left, int right) {
    int pivot = array[right];
    int i = left - 1;

    for (int j = left; j < right; ++j) {
        if (array[j] <= pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }

    swap(&array[i + 1], &array[right]);
    return i + 1;
}

void quicksort(int* array, int left, int right) {
    if (left < right) {
		int pivot = partition(array, left, right);
		#pragma omp task
        {
            //printf("Thread %d sorting left partition\n", omp_get_thread_num());
            quicksort(array, left, pivot - 1);
        }

		#pragma omp task
        {
            //printf("Thread %d sorting right partition\n", omp_get_thread_num());
            quicksort(array, pivot + 1, right);
        }

		#pragma omp taskwait
    }
}

int main(int argc, char** argv) {
    int size = atoi(argv[1]);
    int* arr = generate_random_array(size);

    for (int i = 0; i < size; i++) {
        if (LOG) printf("%d->", arr[i]);
    }
    printf("\nAfter sorting:\n");

	#pragma omp parallel
    quicksort(arr, 0, size - 1);
	

    for (int i = 0; i < size; i++) {
        if (LOG) printf("%d->", arr[i]);
    }
    if (DEBUG && !check(arr, size)) {
        printf("Array is not sorted");
    }

    // free memory
    free(arr);
}
