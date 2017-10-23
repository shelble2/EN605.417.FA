/**
 * Assignment 08 B
 *
 * Uses the cuBLAS library to invert a matrix
 *
 * Sarah Helble
 * 22 Oct 2017
 *
 **/

#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include <random_square_grid.h>

#include <cuda.h>
#include <cublas.h>

#define MAX_INT 9               // 9 is standard sudoku
#define CELLS MAX_INT * MAX_INT // sudokus are square 9 x 9

#define THREADS_PER_BLOCK 1

/**
 * This kernel inverts the passed matrix
 * TODO: add more description
 */
__global__ void invert_matrix(unsigned int* matrix) {
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //TODO: do the inversion
}

void main_sub( ) {
  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;

  curandState_t* states;
  unsigned int* nums, *d_nums;

  cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  cudaMallocHost((void**) &nums, CELLS * sizeof(unsigned int));
  cudaMalloc((void**) &states, CELLS * sizeof(curandState_t));
  cudaMalloc((void**) &d_nums, CELLS * sizeof(unsigned int));

  /* Recording from init to copy back */
	cudaEventRecord(start, 0);

  /* Allocate space and invoke the GPU to initialize the states for cuRAND */
  init_states<<<num_blocks, num_threads>>>(time(0), states);

  /* invoke the kernel to generate random numbers */
  fill_grid<<<num_blocks, num_threads>>>(states, d_nums);

  /* copy the result back to the CPU */
  cudaMemcpy(nums, d_nums, CELLS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);
  printf("Elapsed Time: %f", duration);

  sudoku_print(nums);

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  cudaFree(d_nums);
  cudaFree(nums);
}

/**
 * Starting here so that we can easily execute two runs of each kernel without
 * modifying surrounding functions
 */
int main() {
  int iters = 3;

  for(int i = 0; i < iters; i++) {
    printf("\nRun #%d of kernel function:\n", i+1);
    main_sub();
    printf("\n");
  }

  return 0;
}
