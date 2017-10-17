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
#include <stdbool.h>

#define NUM_ELEMENTS 512
#define THREADS_PER_BLOCK 256

#define MAX_INT 30

/**
 * Returns the current time
 */
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

/**
 * Kernel function that takes a moving average of the values in
 * @list and puts the results in @averages
 * Uses registers to store the calculations.
 */
__global__ void average_using_registers(unsigned int *list, float *averages)
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
 * Kernel function that takes a moving average of the values in
 * @list and puts the results in @averages
 * Uses shared memory to store the calculations.
 */
__global__ void average_using_shared(unsigned int *list, float *averages)
{
	__shared__ unsigned int sums[NUM_ELEMENTS];
	__shared__ unsigned int nums[NUM_ELEMENTS];

	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(idx < NUM_ELEMENTS) {
		sums[idx] = list[idx];
		nums[idx] = 1;

		// If there is a previous element, add it to sum
		if(idx > 0) {
			sums[idx] = sums[idx] + list[idx - 1];
			nums[idx] = nums[idx] + 1;
		}

		// If there is a next element, add it to sum
		if((idx + 1) < NUM_ELEMENTS) {
			sums[idx] = sums[idx] + list[idx + 1];
			nums[idx] = nums[idx] + 1;
		}

		// Calculate the average
		averages[idx] = (float) sums[idx] / nums[idx];
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
 * @use_registers is 1 if registers should be used. Otherwise, will call
 * kernel that uses shared memory
 */
void exec_kernel(bool use_registers)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (NUM_ELEMENTS));
  int float_array_size_in_bytes = (sizeof(float) * (NUM_ELEMENTS));
  int i = 0;

  unsigned int *list;
  float *averages;

  //pin it
  cudaMallocHost((void **)&list, array_size_in_bytes);
  cudaMallocHost((void **)&averages, float_array_size_in_bytes);

	// Fill array with random numbers between 0 and MAX_INT
  for(i = 0; i < NUM_ELEMENTS; i++) {
  	list[i] = (unsigned int) rand() % MAX_INT;
  }

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *d_list;
  float *d_averages;

  cudaMalloc((void **)&d_list, array_size_in_bytes);
  cudaMalloc((void **)&d_averages, float_array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_list, list, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = NUM_ELEMENTS/THREADS_PER_BLOCK;
  const unsigned int num_threads = NUM_ELEMENTS/num_blocks;

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;

  cudaEvent_t start_time = get_time();

	if(use_registers) {
		average_using_registers<<<num_blocks, num_threads>>>(d_list, d_averages);
	} else {
		average_using_shared<<<num_blocks, num_threads>>>(d_list, d_averages);
	}

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);

  cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( averages, d_averages, float_array_size_in_bytes, cudaMemcpyDeviceToHost);

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
	printf("First Run of Averages Calculated using Shared Memory");
  exec_kernel(false);
	printf("-----------------------------------------------------------------\n");

	printf("Second Run of Averages Calculated using Shared Memory");
  exec_kernel(false);
	printf("-----------------------------------------------------------------\n");

  /* Do the average with registers*/
	printf("First Run of Averages Calculated using Register Memory");
  exec_kernel(true);
  printf("-----------------------------------------------------------------\n");

	printf("Second Run of Averages Calculated using Register Memory");
  exec_kernel(true);
  printf("-----------------------------------------------------------------\n");



  return EXIT_SUCCESS;
}
