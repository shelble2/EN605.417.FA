/**
 * Assignment 05 Program
 * Sarah Helble
 * 9/29/17
 *
 * Usage ./out <total_num_threads> <threads_per_block>
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

/*
 * Shuffles the passed @ordered list and places the result in @shuffled
*/
//TODO: follow the 'plan'
__global__ void shuffle(unsigned int *ordered; unsigned int *shuffled)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  const unsigned int instruction_number = idx % 20;

  const int instruction = 0;//plan[instruction_number];

  shuffled[idx + instruction] = ordered[idx];
}

/* Sets up all of the memory and arrays for
 * call to Kernel
 */
void main_sub(int num_threads, int threads_per_block)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (array_size));

  //host pinned
  unsigned int *ordered;
  unsigned int *shuffled_result;

  //pin it
  cudaMallocHost((void **)&ordered, array_size_in_bytes);
  cudaMallocHost((void **)&shuffled_result, array_size_in_bytes);

  //Put values in the ordered array
  for(int i = 0; i < num_threads; i++) {
    ordered[i] = i;
  }

  unsigned int *d_ordered;
  unsigned int *d_shuffled_result;

  cudaMalloc((void **)&d_ordered, array_size_in_bytes);
  cudaMalloc((void **)&d_shuffled_result, array_size_in_bytes);

  //Copy host mem to device
  cudaMemcpy( d_ordered, ordered, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = array_size/threads_per_block;
  const unsigned int num_threads = array_size/num_blocks;

  shuffle<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);

  //copy the result back
  cudaMemcpy( shuffled_result, d_shuffled_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  //print
  for(int i = 0; i < num_threads; i++) {
      printf("original: %d, shuffled: %d", ordered[i], shuffled_result[i]);
  }

  /* Free the GPU memory */
  cudaFree(d_ordered);
  cudaFree(d_shuffled_result);

  // Free the host memory
  cudaFreeHost(ordered);
  cudaFreeHost(shuffled_result);
}

/**
 * Entry point for excution. Checks command line arguments and
 * opens input files, then passes execution to subordinate main_sub()
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

  main_sub(num_threads, threads_per_block);
}
