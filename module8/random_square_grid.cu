/**
 * Assignment 08
 * Beginnings of what would be needed to produce random sudoku puzzles.
 * This program produces a square matrix and fills each cell with a random
 * value between 1 and MAX_INT (inclusive).
 *
 * In order to try out a second CUDA library, this program also uses cuBLAS to
 * invert the resulting matrix
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
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX_INT 9               // 9 is standard sudoku
#define CELLS MAX_INT * MAX_INT // sudokus are square 9 x 9

#define THREADS_PER_BLOCK 1

/**
 * This kernel initializes the states for each input in the array
 * @seed is the seed for the init function
 * @states is an allocated array, where the output of this fuction will be stored
 */
__global__ void init_states(unsigned int seed, curandState_t* states) {
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
__global__ void fill_grid(curandState_t* states, unsigned int* numbers) {
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  numbers[idx] = curand(&states[idx]) % MAX_INT;

  // If we got a 0, make it MAX_INT, since sudokus don't have 0's
  //TODO: think of a more elegant way to do this.
  if(numbers[idx] == 0) {
      numbers[idx] = MAX_INT;
  }
}

/**
 * Multiply the two passed matrices into the resultant matrix
 * @A and @B are the matrices to Multiply
 * @result is the result
 */
__global__ void matrix_multiply(unsigned int *A, unsigned int *B, unsigned int *result) {
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  //TODO: actually multiply here instead of just copying A
  result[idx] = A[idx];
}

/**
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
 void sudoku_print(unsigned int* numbers)
{
  int i;
  int j;
  int block_dim = round(sqrt(MAX_INT));

  printf("\n_________________________________________\n");

  for (i = 0; i < MAX_INT; i++) {

    printf("||");
    for (j = 0; j < MAX_INT; j++) {
      printf(" %u |", numbers[ ( (i*MAX_INT) + j ) ]);
      if((j+1) % block_dim == 0) {
          printf("|");
      }
    }

    j = 0;
    //Breaks between each row
    if( ((i+1) % block_dim) == 0) {
      printf("\n||___|___|___||___|___|___||___|___|___||\n");
    } else {
      //TODO:make this able to handle other sizes prettily
      printf("\n||---|---|---||---|---|---||---|---|---||\n");
    }
 }
}

/**
 * Harness for the creation of random nxn matrices
 * Returns matrix of unsigned ints to be freed by caller
 */
void rand_sub(unsigned int **out) {
  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;

  curandState_t* states;
  unsigned int *tmp, *nums, *d_nums;

  cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  tmp = (unsigned int*)malloc(CELLS * (sizeof(unsigned int)));
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

  memcpy(tmp, nums, CELLS *sizeof(unsigned int));

  cudaFreeHost(nums);

  *out = tmp;
}

/**
 * Sub function for handling calls to the matrix multiplication
 * kernel function
 * @matrix_1 and @matrix_2 are two matrices to multiply together
 */
void blas_sub(unsigned int *matrix_1, unsigned int *matrix_2)
{
  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;
  const unsigned int array_size_in_bytes = CELLS *sizeof(unsigned int);

  unsigned int *m_A, *m_B, *result, *d_A, *d_B, *d_result;

  cudaEvent_t start, stop;
  float duration;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMallocHost((void**) &m_A, array_size_in_bytes);
  cudaMallocHost((void**) &m_B, array_size_in_bytes);
  cudaMallocHost((void**) &result, array_size_in_bytes);

  //Copy passed arrays to pinned memory
  memcpy(m_A, matrix_1, array_size_in_bytes);
  memcpy(m_B, matrix_2, array_size_in_bytes);

  cudaMalloc((void**) &d_A, array_size_in_bytes);
  cudaMalloc((void**) &d_B, array_size_in_bytes);
  cudaMalloc((void**) &d_result, array_size_in_bytes);

  //copy pinned host memory to device memory
  cudaMemcpy( d_A, m_A, array_size_in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_B, m_B, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Recording from init to copy back */
  cudaEventRecord(start, 0);

  /* Allocate space and invoke the GPU to initialize the states for cuRAND */
  matrix_multiply<<<num_blocks, num_threads>>>(d_A, d_B, d_result);

  //Copy the result back to the host
  cudaMemcpy(result, d_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);
  printf("Elapsed Time: %f", duration);

  sudoku_print(result);

  cudaFreeHost(m_A);
  cudaFreeHost(m_B);
  cudaFreeHost(result);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_result);
}

/**
 * Starting here so that we can easily execute two runs of each kernel without
 * modifying surrounding functions
 */
int main() {
  unsigned int *A, *B, *C, *D;

  printf("\nRun #1 of cuRAND kernel function. Matrix A:\n");
  rand_sub(&A);
  printf("\n");

  printf("\nRun #2 of cuRAND kernel function. Matrix B:\n");
  rand_sub(&B);
  printf("\n");

  printf("\nRun #3 of cuRAND kernel function. Matrix C:\n");
  rand_sub(&C);
  printf("\n");

  printf("\nRun #4 of cuRAND kernel function. Matrix D:\n");
  rand_sub(&D);
  printf("\n");

  printf("\nRun #1 of cuBLAS kernel function. Matrix A x Matrix B:\n");
  blas_sub(A, B);

  printf("\nRun #2 of cuBLAS kernel function. Matrix C x Matrix D:\n");
  blas_sub(C, D);

  free(A);
  free(B);
  free(C);
  free(D);
  return 0;
}
