/**
 * Assignment 03 Program - ceasar_cipher.cu
 * Sarah Helble
 * 9/17/17
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
 */

#include <stdio.h>
#include <stdlib.h>

/* Number of characters in the alphabet */
#define NUM_ALPHA 127

/**
 * Kernel function that creates a ciphertext by adding the values
 * in @text to the values in @key. As in a caesar cipher with keyword.
 *
 * @text plaintext values
 * @key key values
 * @result ciphertext
 *
 * TODO: some of the values in the resultant ciphertext are unprintable.
 * Make wrap around more advanced to deal with this.
 */
__global__ void encrypt(unsigned int *text, unsigned int *key, unsigned int *result, unsigned int *thread, unsigned int *block)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  /* Create the cipherchar (addition of plaintext char and key char */
  result[idx] = (unsigned int) ( ( key[idx] + text[idx] ) % NUM_ALPHA );

  /* Calculating these extras so that we can see which blocks/threads do what */
  thread[idx] = threadIdx.x;
  block[idx] = blockIdx.x;

  /* TODO: Some of these values are unprintable. Make wrap more advanced */
}

/**
 * One fuction to handle the printing of all results.
 * @text is the plaintext array
 * @key is the key used to encrypt
 * @result is the resulting ciphertext
 * @blocks is the array holding the block number for each calculation
 * @threads is the array holding the thread number fo each calculation
 */
void print_all_results(unsigned int *text, unsigned int *key, unsigned int *result, unsigned int *blocks, unsigned int *threads, int array_size)
{
  int i;

  /* Print the calculations */
  for(i = 0; i < array_size; i++) {
    printf("Block %2u - Thread %2u Calculated: %c + %c = %c\n", blocks[i], threads[i], text[i], key[i], result[i]);
  }

  /* Print the plain text, key, and result */
  printf("\nSummary:\n\nEncrypted text: ");
  for(i = 0; i < array_size; i++) {
    printf("%c", text[i]);
  }
  printf("\n\nWith Key: ");
  for(i = 0; i < array_size; i++) {
    printf("%c", key[i]);
  }
  printf("\n\nResults in ciphertext: ");
  for(i = 0; i < array_size; i++) {
    printf("%c", result[i]);
  }
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
void main_sub(int array_size, int threads_per_block, FILE *input_fp, FILE *key_fp)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
  int i = 0;

  unsigned int cpu_text[array_size];
  unsigned int cpu_key[array_size];
  unsigned int cpu_result[array_size];
  unsigned int cpu_threads[array_size];
  unsigned int cpu_blocks[array_size];

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < array_size; i++) {
    cpu_text[i] = fgetc(input_fp);
    cpu_key[i] = fgetc(key_fp);
  }

  /* Close the file pointers */
  fclose(input_fp);
  fclose(key_fp);

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *gpu_text;
  unsigned int *gpu_key;
  unsigned int *gpu_result;
  unsigned int *gpu_threads;
  unsigned int *gpu_blocks;
  cudaMalloc((void **)&gpu_text, array_size_in_bytes);
  cudaMalloc((void **)&gpu_key, array_size_in_bytes);
  cudaMalloc((void **)&gpu_result, array_size_in_bytes);
  cudaMalloc((void **)&gpu_threads, array_size_in_bytes);
  cudaMalloc((void **)&gpu_blocks, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  /* Execute the encryption kernel */
  encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result, gpu_threads, gpu_blocks);

  /* Copy the GPU memory back to the CPU */
  cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( cpu_threads, gpu_threads, array_size_in_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( cpu_blocks, gpu_blocks, array_size_in_bytes, cudaMemcpyDeviceToHost);

  /* Free the GPU memory */
  cudaFree(gpu_text);
  cudaFree(gpu_key);
  cudaFree(gpu_result);
  cudaFree(gpu_threads);
  cudaFree(gpu_blocks);

  print_all_results(cpu_text, cpu_key, cpu_result, cpu_blocks, cpu_threads, array_size);
}

int main(int argc, char *argv[])
{
  /* Check the number of arguments, print usage if wrong */
  if(argc != 5) {
    printf("Error, usage: %s <total_num_threads> <threads_per_block> <input_file> <key_file>\n", argv[0]);
    exit(-1);
  }

  int num_threads = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);
  char *input_filename = argv[3];
  char *key_filename = argv[4];

  /* Make sure the input text file and the key file are openable */
  FILE *input_fp = fopen(input_filename, "r");
  if(!input_fp) {
    printf("Error: failed to open input file %s\n", argv[3]);
    exit(-1);
  }
  FILE *key_fp = fopen(key_filename, "r");
  if(!key_fp){
    printf("Error: failed to open key file %s\n", argv[4]);
    fclose(input_fp);
    exit(-1);
  }

  /* Pass all arguments to the subordinate main function */
  main_sub(num_threads, threads_per_block, input_fp, key_fp);

  return EXIT_SUCCESS;
}
