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
void pageable_transfer_execution(int array_size, int threads_per_block)
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
 * Function that sets up everything for the kernel function encrypt()
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 * @input_fp file pointer to the input file text
 * @key_fp file pointer to the key file
 *
 * Closes the file pointers @input_fp and @key_fp
 */
void pinned_transfer_execution(int array_size, int threads_per_block, FILE *input_fp, FILE *key_fp)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
  int i = 0;

  /*host pageable */
  unsigned int *cpu_text_pageable = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_key_pageable = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_result_pageable = (unsigned int *) malloc(array_size_in_bytes);

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < array_size; i++) {
    cpu_text_pageable[i] = fgetc(input_fp);
    cpu_key_pageable[i] = fgetc(key_fp);
    if(feof(input_fp) || feof(key_fp)) {
        rewind(input_fp);
        rewind(key_fp);
    }
  }

  //host pinned
  unsigned int *cpu_text_pinned;
  unsigned int *cpu_key_pinned;
  unsigned int *cpu_result_pinned;

  //pin it
  cudaMallocHost((void **)&cpu_text_pinned, array_size_in_bytes);
  cudaMallocHost((void **)&cpu_key_pinned, array_size_in_bytes);
  cudaMallocHost((void **)&cpu_result_pinned, array_size_in_bytes);

  /* Copy the memory over */
  memcpy(cpu_text_pinned, cpu_text_pageable, array_size_in_bytes);
  memcpy(cpu_key_pinned, cpu_key_pageable, array_size_in_bytes);
  memcpy(cpu_result_pinned, cpu_result_pageable, array_size_in_bytes);

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *gpu_text;
  unsigned int *gpu_key;
  unsigned int *gpu_result;

  cudaMalloc((void **)&gpu_text, array_size_in_bytes);
  cudaMalloc((void **)&gpu_key, array_size_in_bytes);
  cudaMalloc((void **)&gpu_result, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy( gpu_text, cpu_text_pinned, array_size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( gpu_key, cpu_key_pinned, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  /* Execute the encryption kernel and keep track of start and end time for duration */
  float duration = 0;
  cudaEvent_t start_time = get_time();

  shuffle<<<num_blocks, num_threads>>>(gpu_text, gpu_result);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( cpu_result_pinned, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Pinned Transfer- Duration: %fmsn\n", duration);
  print_results(cpu_text_pinned, cpu_result_pinned, array_size);

  /* Free the GPU memory */
  cudaFree(gpu_text);
  cudaFree(gpu_key);
  cudaFree(gpu_result);

  /* Free the pinned CPU memory */
  cudaFreeHost(cpu_text_pinned);
  cudaFreeHost(cpu_key_pinned);
  cudaFreeHost(cpu_result_pinned);

  /* Free the pageable CPU memory */
  free(cpu_text_pageable);
  free(cpu_key_pageable);
  free(cpu_result_pageable);

}

/**
 * Prints the correct usage of this file
 * @name is the name of the executable (argv[0])
 */
void print_usage(char *name)
{
  printf("Usage: %s <total_num_threads> <threads_per_block> <input_file> <key_file>\n", name);
}

/**
 * Performs simple setup functions before calling the pageable_transfer_execution()
 * function.
 * Makes sure the files are valid, handles opening and closing of file pointers.
 */
void pageable_transfer(int num_threads, int threads_per_block, char *input_file, char *key_file)
{
  /* Make sure the input text file and the key file are openable */
  FILE *input_fp = fopen(input_file, "r");
  if(!input_fp) {
    printf("Error: failed to open input file %s\n", input_file);
    exit(-1);
  }
  FILE *key_fp = fopen(key_file, "r");
  if(!key_fp){
    printf("Error: failed to open key file %s\n", key_file);
    fclose(input_fp);
    exit(-1);
  }

  /* Perform the pageable transfer */
  pageable_transfer_execution(num_threads, threads_per_block);

  fclose(input_fp);
  fclose(key_fp);
}

/**
 * Performs setup functions before calling the pageable_transfer_execution()
 * function.
 * Makes sure the files are valid, handles opening and closing of file pointers.
 */
void pinned_transfer(int num_threads, int threads_per_block, char *input_file, char *key_file)
{
  /* Make sure the input text file and the key file are openable */
  FILE *input_fp = fopen(input_file, "r");
  if(!input_fp) {
    printf("Error: failed to open input file %s\n", input_file);
    exit(-1);
  }
  FILE *key_fp = fopen(key_file, "r");
  if(!key_fp){
    printf("Error: failed to open key file %s\n", key_file);
    fclose(input_fp);
    exit(-1);
  }

  /* Perform the pageable transfer */
  pinned_transfer_execution(num_threads, threads_per_block, input_fp, key_fp);

  fclose(input_fp);
  fclose(key_fp);
}

/**
 * Entry point for excution. Checks command line arguments and
 * opens input files, then passes execution to subordinate main_sub()
 */
int main(int argc, char *argv[])
{
  /* Check the number of arguments, print usage if wrong */
  if(argc != 5) {
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
  pageable_transfer(num_threads, threads_per_block, argv[3], argv[4]);

  printf("-----------------------------------------------------------------\n");

  /* Perform the pinned transfer */
  pinned_transfer(num_threads, threads_per_block, argv[3], argv[4]);

  return EXIT_SUCCESS;
}
