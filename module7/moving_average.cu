threads_per_block/**
 * Assignment 07 Program - moving_average.cu  (edited from module 6 for 7)
 * Sarah Helble
 * 10/16/17
 *
 * Calculates the average of each index and its neighbors
 *
 * Usage ./a.out [-v] [-n num_elements] [-b threads_per_block] [-m max_int]
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <unistd.h>

#define DEFAULT_NUM_ELEMENTS 512
#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_MAX_INT 30

/**
 * Kernel function that takes a moving average of the values in
 * @list and puts the results in @averages
 * Uses registers to store the calculations.
 */
__global__ void average_window(unsigned int *list, float *averages, int num_elements)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(idx < num_elements) {
		unsigned int sum = list[idx];
		unsigned int num = 1;

		// If there is a previous element, add it to sum
		if(idx > 0) {
			sum = sum + list[idx - 1];
			num = num + 1;
		}

		// If there is a next element, add it to sum
		if((idx + 1) < num_elements) {
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
  for(i = 0; i < num_elements; i++) {
    printf("Original value at index [%d]: %d, average: %f\n", i, list[i], averages[i]);
  }
  printf("\n");
}

/**
 * Function that sets up everything for the kernel function
 *
 * @verbosity is 1 if the function should print detailed results of averages
 * verbosity of 0 will only print timing data
 */
void exec_kernel_sync(int verbosity, int num_elements, int threads_per_block, int max_int)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (num_elements));
  int float_array_size_in_bytes = (sizeof(float) * (num_elements));
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
  for(i = 0; i < num_elements; i++) {
  	list[i] = (unsigned int) rand() % max_int;
  }

	/* Recording from copy to copy back */
	cudaEventRecord(start, 0);

	/* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_list, list, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = num_elements/threads_per_block;
  const unsigned int num_threads = num_elements/num_blocks;

	/* Kernel call */
	average_window<<<num_blocks, num_threads>>>(d_list, d_averages, num_elements);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( averages, d_averages, float_array_size_in_bytes, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  printf("\tList size: %d, Duration: %fmsn\n", num_elements, duration);
  if(verbosity) {
    print_results(list, averages);
  }

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
 * @verbosity is 1 if the function should print detailed results of averages
 * verbosity of 0 will only print timing data
 */
void exec_kernel_async(int verbosity, int num_elements, int threads_per_block, int max_int)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (num_elements));
  int float_array_size_in_bytes = (sizeof(float) * (num_elements));
  int i = 0;

  unsigned int *list, *d_list;
  float *averages, *d_averages;

	cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

	cudaMalloc((void **)&d_list, array_size_in_bytes);
  cudaMalloc((void **)&d_averages, float_array_size_in_bytes);

  cudaMallocHost((void **)&list, array_size_in_bytes);
  cudaMallocHost((void **)&averages, float_array_size_in_bytes);

	// Fill array with random numbers between 0 and MAX_INT
  for(i = 0; i < num_elements; i++) {
  	list[i] = (unsigned int) rand() % max_int;
  }

	/* Recording from copy to copy back */
	cudaEventRecord(start, 0);

	/* Copy the CPU memory to the GPU memory asynchronously */
  cudaMemcpyAsync(d_list, list, array_size_in_bytes, cudaMemcpyHostToDevice, stream);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = num_elements/threads_per_block;
  const unsigned int num_threads = num_elements/num_blocks;

	/* Kernel call */
	average_window<<<num_blocks, num_threads>>>(d_list, d_averages, num_elements);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpyAsync( averages, d_averages, float_array_size_in_bytes, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  printf("\tList size: %d, Duration: %fmsn\n", num_elements, duration);
  if(verbosity) {
    print_results(list, averages);
  }

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
  int verbosity = 0;
  int num_elements      = DEFAULT_NUM_ELEMENTS;
  int threads_per_block = DEFAULT_THREADS_PER_BLOCK;
  int max_int           = DEFAULT_MAX_INT;
  int c;

  while((c = getopt(argc, argv, "vn:b:m:")) != -1) {
    switch(c) {
      case 'v':
        verbosity = 1;
        break;
      case 'n':
        num_elements = atoi(optarg);
        break;
      case 'b':
        threads_per_block = atoi(optarg);
        break;
      case 'm':
        max_int = atoi(optarg);
        break;
      default:
        printf("Error: unrecognized option: %c\n", c);
        printf("Usage: %s [-v] [-n num_elements] [-b threads_per_block] [-m max_int]", argv[0]);
        exit(-1);
      }
  }
  printf("verbosity: %d\tnum_elements: %d\tthreads_per_block: %d\tmax_int: %d\n",
    verbosity, num_elements, threads_per_block, max_int);

	/* Do the average with shared memory */
	printf("\nFirst Run of Averages done synchronously");
  exec_kernel_sync(verbosity, num_elements, threads_per_block, max_int);
	printf("-----------------------------------------------------------------\n");

	printf("Second Run of Averages done asynchronously");
  exec_kernel_async(verbosity, num_elements, threads_per_block, max_int);
	printf("-----------------------------------------------------------------\n");

  return EXIT_SUCCESS;
}
