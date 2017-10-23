/**
 * Assignment 08
 * Beginnings of what would be needed to produce random sudoku puzzles.
 * This program produces a square matrix and fills each cell with a random
 * value between 0 and MAX_INT.
 *
 * Future work will make this production follow the rules of sudoku (i.e., one
 * of each value in each row, col, square)
 *
 * Sarah Helble
 * 22 Oct 2017
 *
 **/

#include <unistd.h>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX_INT 9                       // 9 is standard sudoku
#define CELLS (MAX_INT+1) * (MAX_INT+1) // sudokus are square 10 x 10

#define THREADS_PER_BLOCK 1

/**
 * This kernel initializes the states for each input in the array
 * @seed is the seed for the init function
 * @states is an allocated array, where the output of this fuction will be stored
 */
__global__ void init(unsigned int seed, curandState_t* states) {
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Given the passed array of states @states, this kernel fills allocated array
 * @numbers with a random int between 0 and MAX_INT
 * @states is the set of states already initialized by CUDA
 * @numbers is an allocated array where this kernel function will put its output
 */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  numbers[idx] = curand(&states[idx]) % MAX_INT;
}

/**
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
 //TODO: need to add block breaks
 void sudoku_print(unsigned int* numbers)
{
  int i;
  int j;

  printf("\n________________________________________________________\n")

  for (i = 0; i <= MAX_INT; i++) {
    for (j = 0; j <= MAX_INT; j++) {
      printf("| %u |", numbers[ ( (i*MAX_INT) + j ) ]);
    }
    printf("\n------------------------------------------------------\n")
    j = 0;
  }

  printf("\n________________________________________________________\n")
}

int main( ) {
  //TODO: make it print like a sudoku. Put it in 2D array

   /* CUDA's random number library uses curandState_t to keep track
      of the seed value
      we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, CELLS * sizeof(curandState_t));

  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;

  /* invoke the GPU to initialize all of the random states */
  init<<<num_blocks, num_threads>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  unsigned int cpu_nums[CELLS];
  unsigned int* gpu_nums;
  cudaMalloc((void**) &gpu_nums, CELLS * sizeof(unsigned int));

  /* invoke the kernel to get some random numbers */
  randoms<<<num_blocks, num_threads>>>(states, gpu_nums);

  /* copy the random numbers back */
  cudaMemcpy(cpu_nums, gpu_nums, CELLS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  /* print them out */
  for (int i = 0; i < CELLS; i++) {
    printf("%u\n", cpu_nums[i]);
  }

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(gpu_nums);

  return 0;
}
