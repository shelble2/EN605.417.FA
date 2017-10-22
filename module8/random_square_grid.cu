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

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, unsigned int* numbers) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = curand(&states[blockIdx.x]) % MAX_INT;
}

int main( ) {
  //TODO: Clean this up, make it print like a sudoku. Put it in array
  // Can also do multiple threads per block

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
