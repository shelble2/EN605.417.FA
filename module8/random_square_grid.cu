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
#include <cublas.h>

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
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
 void sudoku_print_float(float* numbers)
 {
  int i;
  int j;
  int block_dim = round(sqrt(MAX_INT));

  printf("\n_________________________________________\n");

  for (i = 0; i < MAX_INT; i++) {

    printf("||");
    for (j = 0; j < MAX_INT; j++) {
      printf(" %f |", numbers[ ( (i*MAX_INT) + j ) ]);
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

  //TODO: goes too fast; all get same seed. Only has second resolution
  unsigned int seed = time(NULL);
  printf("seed: %d ", seed);

  /* Recording from init to copy back */
	cudaEventRecord(start, 0);

  /* Allocate space and invoke the GPU to initialize the states for cuRAND */
  init_states<<<num_blocks, num_threads>>>(seed, states);

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
int blas_sub(unsigned int *matrix_1, unsigned int *matrix_2)
{
  const unsigned int array_size_in_bytes = CELLS *sizeof(float);
  float *h_m1, *h_m2, *h_result, *d_m1, *d_m2, *d_result;

  cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  //Get the passed arrays into pinned host memory
  // And need to convert to float for cublas
  cudaMallocHost((void**) &h_m1, array_size_in_bytes);
  cudaMallocHost((void**) &h_m2, array_size_in_bytes);

  for(int i = 0; i < CELLS; i++){
      h_m1[i] = (float)matrix_1[i];
      h_m2[i] = (float)matrix_2[i];
  }

  cublasStatus status;
  cublasInit();

  // We know its of size CELLS because both arrays are guaranteed to be
  // the same size and square
  h_result = (float *)malloc(array_size_in_bytes);
  if(h_result == NULL) {
      printf("Error: failed to allocate memory for result array\n");
      return EXIT_FAILURE;
  }

  /* Recording from cublasAllocs to copy back */
	cudaEventRecord(start, 0);

  // Allocate device memory
  status = cublasAlloc(CELLS, sizeof(float), (void **)&d_m1);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: failed to allocate device memory for m1\n");
    return EXIT_FAILURE;
  }

  status = cublasAlloc(CELLS, sizeof(float), (void**)&d_m2);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: failed to allocate device memory for m2\n");
    return EXIT_FAILURE;
  }

  status = cublasAlloc(CELLS, sizeof(float), (void**)&d_result);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: failed to allocate device memory for result\n");
    return EXIT_FAILURE;
  }

  // Set input matrices
  status = cublasSetMatrix(MAX_INT, MAX_INT, sizeof(float), h_m1, MAX_INT, d_m1, MAX_INT);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: failed to set matrix 1\n");
    return EXIT_FAILURE;
  }

  status = cublasSetMatrix(MAX_INT, MAX_INT, sizeof(float), h_m2, MAX_INT, d_m2, MAX_INT);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: failed to set matrix 2\n");
    return EXIT_FAILURE;
  }
  //TODO: add timing data

  //run the kernel
  cublasSgemm('n', 'n', MAX_INT, MAX_INT, MAX_INT, 1, d_m1, MAX_INT, d_m2, MAX_INT, 0, d_result, MAX_INT);

  status = cublasGetError();
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: cublas kernel returned failure");
    return EXIT_FAILURE;
  }

  cublasGetMatrix(MAX_INT, MAX_INT, sizeof(float), d_result, MAX_INT, h_result, MAX_INT);
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: cublas get matrix returned failure");
    return EXIT_FAILURE;
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);
  printf("Elapsed Time: %f\n", duration);

  for(int i = 0; i < CELLS; i++) {
      printf("%f\n", h_result[i]);
  }
  //sudoku_print(h_result);

  cublasFree(d_m1);
  cublasFree(d_m2);
  cublasFree(d_result);

  cudaFreeHost(h_m1);
  cudaFreeHost(h_m2);
  free(h_result);

  status = cublasShutdown();
  if(status != CUBLAS_STATUS_SUCCESS) {
    printf("Error: cublas shutdown returned failure");
    return EXIT_FAILURE;
  }
 return EXIT_SUCCESS;
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
