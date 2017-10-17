/**
 * Assignment 07 Program - moving_average.cu  (edited from module 6 for 7)
 * Sarah Helble
 * 10/16/17
 *
 * Calculates the average of each index and its neighbors
 *
 * Usage ./aout
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define NUM_ELEMENTS 512
#define THREADS_PER_BLOCK 256

#define MAX_INT 30

/**
 * Kernel function that takes a moving average of the values in
 * @list and puts the results in @averages
 * Uses registers to store the calculations.
 */
__global__ void average_window(unsigned int *list, float *averages)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(idx < NUM_ELEMENTS) {
		unsigned int sum = list[idx];
		unsigned int num = 1;

		// If there is a previous element, add it to sum
		if(idx > 0) {
			sum = sum + list[idx - 1];
			num = num + 1;
		}

		// If there is a next element, add it to sum
		if((idx + 1) < NUM_ELEMENTS) {
			sum = sum + list[idx + 1];
			num = num + 1;
		}

		averages[idx] = (float) sum / num;
	}
}

/**
 * Fuction to handle the printing of results.
 * @list is the original array
 * @averages is the result
 */
void print_results(unsigned int *list, float *averages)
{
  int i = 0;

  printf("\n");
  for(i = 0; i < NUM_ELEMENTS; i++) {
    printf("Original value at index [%d]: %d, average: %f\n", i, list[i], averages[i]);
  }
  printf("\n");
}

/**
 * Function that sets up everything for the kernel function
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 */
void exec_kernel_sync()
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (NUM_ELEMENTS));
  int float_array_size_in_bytes = (sizeof(float) * (NUM_ELEMENTS));
  int i = 0;

  unsigned int *list, *d_list;
  float *averages, *d_averages;

	cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void **)&d_list, array_size_in_bytes);
  cudaMalloc((void **)&d_averages, float_array_size_in_bytes);

  cudaMallocHost((void **)&list, array_size_in_bytes);
  cudaMallocHost((void **)&averages, float_array_size_in_bytes);

	// Fill array with random numbers between 0 and MAX_INT
  for(i = 0; i < NUM_ELEMENTS; i++) {
  	list[i] = (unsigned int) rand() % MAX_INT;
  }

	/* Recording from copy to copy back */
	cudaEventRecord(start, 0);

	/* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_list, list, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = NUM_ELEMENTS/THREADS_PER_BLOCK;
  const unsigned int num_threads = NUM_ELEMENTS/num_blocks;

	/* Kernel call */
	average_window<<<num_blocks, num_threads>>>(d_list, d_averages);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( averages, d_averages, float_array_size_in_bytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  printf("\tDuration: %fmsn\n", duration);
  print_results(list, averages);

  /* Free the GPU memory */
  cudaFree(d_list);
  cudaFree(d_averages);

  /* Free the pinned CPU memory */
  cudaFreeHost(list);
  cudaFreeHost(averages);
}

/**
 * Function that sets up everything for the kernel function
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 */
void exec_kernel_async()
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (NUM_ELEMENTS));
  int float_array_size_in_bytes = (sizeof(float) * (NUM_ELEMENTS));
  int i = 0;

  unsigned int *list, *d_list;
  float *averages, *d_averages;

	cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void **)&d_list, array_size_in_bytes);
  cudaMalloc((void **)&d_averages, float_array_size_in_bytes);

  cudaMallocHost((void **)&list, array_size_in_bytes);
  cudaMallocHost((void **)&averages, float_array_size_in_bytes);

	// Fill array with random numbers between 0 and MAX_INT
  for(i = 0; i < NUM_ELEMENTS; i++) {
  	list[i] = (unsigned int) rand() % MAX_INT;
  }

	/* Recording from copy to copy back */
	cudaEventRecord(start, 0);

	/* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_list, list, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = NUM_ELEMENTS/THREADS_PER_BLOCK;
  const unsigned int num_threads = NUM_ELEMENTS/num_blocks;

	/* Kernel call */
	average_window<<<num_blocks, num_threads>>>(d_list, d_averages);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( averages, d_averages, float_array_size_in_bytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  printf("\tDuration: %fmsn\n", duration);
  print_results(list, averages);

  /* Free the GPU memory */
  cudaFree(d_list);
  cudaFree(d_averages);

  /* Free the pinned CPU memory */
  cudaFreeHost(list);
  cudaFreeHost(averages);
}

/**
 * Entry point for execution. Checks command line arguments
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{

  printf("\n");

	/* Do the average with shared memory */
	printf("First Run of Averages done synchronously");
  exec_kernel_sync();
	printf("-----------------------------------------------------------------\n");

	printf("Second Run of Averages done asynchronously");
  exec_kernel_async();
	printf("-----------------------------------------------------------------\n");

  return EXIT_SUCCESS;
}
