/**
 * Assignment 05 Program 
 * Sarah Helble
 * 9/29/17
 *
 * Usage ./out <total_num_threads> <threads_per_block>
 *
 */

#include <stdio.h>
#include <stdlib.h>

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
 * Kernel function that shuffles the values in @ordered and puts the
 * output in @shuffled
 */
__global__ void shuffle(unsigned int *ordered, unsigned int *shuffled)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  shuffled[idx] = 100;
}

/**
 * One fuction to handle the printing of results.
 * @ordered is the original array
 * @shuffled is the result 
 */
void print_results(unsigned int *ordered, unsigned int *shuffled, int array_size)
{
  int i = 0;

  printf("\n");
  for(i = 0; i < array_size; i++) {
    printf("Original value at index [%d]: %d, shuffled: %d\n", i, ordered[i], shuffled[i]);
  }
  printf("\n");
}

/**
 * Function that sets up everything for the kernel function 
 * with simple pageable host memory
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 */
void pageable_transfer(int array_size, int threads_per_block)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
  int i = 0;

  unsigned int *ordered = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *shuffled_result = (unsigned int *) malloc(array_size_in_bytes);

  // Fill the ordered array
  for(i = 0; i < array_size; i++) {
  	ordered[i] = i;	
  }

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *d_ordered;
  unsigned int *d_shuffled_result;

  cudaMalloc((void **)&d_ordered, array_size_in_bytes);
  cudaMalloc((void **)&d_shuffled_result, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy( d_ordered, ordered, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  /* Execute the encryption kernel and keep track of start and end time for duration */
  float duration = 0;
  cudaEvent_t start_time = get_time();

  shuffle<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( shuffled_result, d_shuffled_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Pageable Transfer- Duration: %fmsn\n", duration);
  print_results(ordered, shuffled_result, array_size);

  /* Free the GPU memory */
  cudaFree(d_ordered);
  cudaFree(d_shuffled_result);

  /* Free the CPU memory */
  free(ordered);
  free(shuffled_result);
}

/**
 * Function that sets up everything for the kernel function 
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 */
void pinned_transfer(int array_size, int threads_per_block)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
  int i = 0;

  /*host pageable */
  unsigned int *ordered_pageable = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *shuffled_result_pageable = (unsigned int *) malloc(array_size_in_bytes);

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < array_size; i++) {
  	ordered_pageable[i] = i;
  }

  //host pinned
  unsigned int *ordered_pinned;
  unsigned int *shuffled_result_pinned;

  //pin it
  cudaMallocHost((void **)&ordered_pinned, array_size_in_bytes);
  cudaMallocHost((void **)&shuffled_result_pinned, array_size_in_bytes);

  /* Copy the memory over */
  memcpy(ordered_pinned, ordered_pageable, array_size_in_bytes);
  memcpy(shuffled_result_pinned, shuffled_result_pageable, array_size_in_bytes);

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *d_ordered;
  unsigned int *d_shuffled_result;

  cudaMalloc((void **)&d_ordered, array_size_in_bytes);
  cudaMalloc((void **)&d_shuffled_result, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_ordered, ordered_pinned, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;
  cudaEvent_t start_time = get_time();

  shuffle<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( shuffled_result_pinned, d_shuffled_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Pinned Transfer- Duration: %fmsn\n", duration);
  print_results(ordered_pinned, shuffled_result_pinned, array_size);

  /* Free the GPU memory */
  cudaFree(d_ordered);
  cudaFree(d_shuffled_result);

  /* Free the pinned CPU memory */
  cudaFreeHost(ordered_pinned);
  cudaFreeHost(shuffled_result_pinned);

  /* Free the pageable CPU memory */
  free(ordered_pageable);
  free(shuffled_result_pageable);
}

/**
 * Prints the correct usage of this file
 * @name is the name of the executable (argv[0])
 */
void print_usage(char *name)
{
  printf("Usage: %s <total_num_threads> <threads_per_block>\n", name);
}

/**
 * Entry point for execution. Checks command line arguments 
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{
  /* Check the number of arguments, print usage if wrong */
  if(argc != 3) {
    printf("Error: Incorrect number of command line arguments\n");
    print_usage(argv[0]);
    exit(-1);
  }

  /* Check the values for num_threads and threads_per_block */
  int num_threads = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);
  if(num_threads <= 0 || threads_per_block <= 0) {
    printf("Error: num_threads and threads_per_block must be integer > 0");
    print_usage(argv[0]);
    exit(-1);
  }

  if(threads_per_block > num_threads) {
      printf("Error: threads per block is greater than number of threads\n");
      print_usage(argv[0]);
      exit(-1);
  }

  printf("\n");
  /* Perform the pageable transfer */
  pageable_transfer(num_threads, threads_per_block);

  printf("-----------------------------------------------------------------\n");

  /* Perform the pinned transfer */
  pinned_transfer(num_threads, threads_per_block);

  return EXIT_SUCCESS;
}
