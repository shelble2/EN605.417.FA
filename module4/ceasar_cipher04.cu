/**
 * Assignment 04 Program - ceasar_cipher.cu expanded to try both pageable and
 * pinned memory
 * Sarah Helble
 * 9/23/17
 *
 * Usage ./out <total_num_threads> <threads_per_block> <input_file> <key_file>
 *
 * Creates two arrays of <total_num_threads> length, and reads <total_num_threads>
 * characters from <input_file> and <key_file> to fill them.
 * Adds the character values together to create a cipher text (caesar cipher with
 * keyword)
 *
 * Uses <total_num_threads> as total number of threads for the execution.
 * Creates blocks with <threads_per_block> each.
 * This results in # blocks = <total_num_threads> / <threads_per_block>
 *
 * Assumes that all values in input_file and key_file are printable (within the
 * range of 32-126 ASCII decimal values)
 */

#include <stdio.h>
#include <stdlib.h>

/*
 * The maximum and minimum integer values of the range of printable characters
 * in the ASCII alphabet. Used by encrypt kernel to wrap adjust values to that
 * ciphertext is always printable.
 */
#define MAX_PRINTABLE 126
#define MIN_PRINTABLE 32
#define NUM_ALPHA MAX_PRINTABLE - MIN_PRINTABLE

/**
 * Kernel function that creates a ciphertext by adding the values
 * in @text to the values in @key. As in a caesar cipher with keyword.
 *
 * @text plaintext values
 * @key key values
 * @result ciphertext
 *
 * TODO: some of the values in the resultant ciphertext will be unprintable.
 * Make wrap around more advanced to deal with this.
 */
__global__ void encrypt(unsigned int *text, unsigned int *key, unsigned int *result)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  /*
   * Adjust value of text and key to be based at 0
   * Printable ASCII starts at MIN_PRINTABLE, but 0 start is easier to work with
   */
  char adjusted_text = text[idx] - MIN_PRINTABLE;
  char adjusted_key = key[idx] - MIN_PRINTABLE;

  /* The cipher character is the text char added to the key char modulo the number of chars in the alphabet*/
  char cipherchar = (adjusted_text + adjusted_key) % (NUM_ALPHA);

  /* adjust back to normal ascii (starting at MIN_PRINTABLE) and save to result */
  result[idx] = (unsigned int) cipherchar + MIN_PRINTABLE ;
}

/**
 * One fuction to handle the printing of all results.
 * @text is the plaintext array
 * @key is the key used to encrypt
 * @result is the resulting ciphertext
 */
void print_all_results(unsigned int *text, unsigned int *key, unsigned int *result, int array_size)
{
  int i = 0;

  /* Print the plain text, key, and result */
  printf("\nSummary:\n\nEncrypted text:\n");
  for(i = 0; i < array_size; i++) {
    printf("%c", text[i]);
  }
  printf("\n\nWith Key:\n");
  for(i = 0; i < array_size; i++) {
    printf("%c", key[i]);
  }
  printf("\n\nResults in ciphertext:\n");
  for(i = 0; i < array_size; i++) {
    printf("%c", result[i]);
  }
  printf("\n\n");
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
void pageable_transfer_execution(int array_size, int threads_per_block, FILE *input_fp, FILE *key_fp)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
  int i = 0;

  unsigned int *cpu_text = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_key = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_result = (unsigned int *) malloc(array_size_in_bytes);

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < array_size; i++) {
    cpu_text[i] = fgetc(input_fp);
    cpu_key[i] = fgetc(key_fp);
  }

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *gpu_text;
  unsigned int *gpu_key;
  unsigned int *gpu_result;

  cudaMalloc((void **)&gpu_text, array_size_in_bytes);
  cudaMalloc((void **)&gpu_key, array_size_in_bytes);
  cudaMalloc((void **)&gpu_result, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  /* Execute the encryption kernel */
  encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  print_all_results(cpu_text, cpu_key, cpu_result, array_size);

  /* Free the GPU memory */
  cudaFree(gpu_text);
  cudaFree(gpu_key);
  cudaFree(gpu_result);

  /* Free the CPU memory */
  free(cpu_text);
  free(cpu_key);
  free(cpu_result);
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

  //TODO: do I need the pageable? or could just do everything from pinned?
  // Something to mention in discussion as well

  /*host pageable */
  unsigned int *cpu_text_pageable = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_key_pageable = (unsigned int *) malloc(array_size_in_bytes);
  unsigned int *cpu_result_pageable = (unsigned int *) malloc(array_size_in_bytes);

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < array_size; i++) {
    cpu_text_pageable[i] = fgetc(input_fp);
    cpu_key_pageable[i] = fgetc(key_fp);
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

  /* Execute the encryption kernel */
  encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( cpu_result_pinned, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  print_all_results(cpu_text_pinned, cpu_key_pinned, cpu_result_pinned, array_size);

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
 * Performs setup functions before calling the pageable_transfer_execution()
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
  pageable_transfer_execution(num_threads, threads_per_block, input_fp, key_fp);

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

  /* Perform the pageable transfer */
  pageable_transfer(num_threads, threads_per_block, argv[3], argv[4]);

  /* Perform the pinned transfer */
  pinned_transfer(num_threads, threads_per_block, argv[3], argv[4]);

  return EXIT_SUCCESS;
}
