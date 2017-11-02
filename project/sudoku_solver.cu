/**
 * sudoku_solver.cu
 * Sarah Helble
 * 2017-11-01
 *
 * In the process of adapting shuffle.cu from Module 5 assigment to work on
 * solving sudoku puzzles in the form of a string of numbers, where 0 indicates
 * an empty cell.
 */

#include <stdio.h>
#include <stdlib.h>

#define MAX_INT 9                 // Customary sudoku
#define CELLS MAX_INT * MAX_INT   // 81
#define THREADS_PER_BLOCK MAX_INT // Seems like a nice way to split..

//TODO: Look up again rules about sharing data across blocks. They will have to share

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
 * Kernel function that moves the values in @ordered to @shuffled
 */
__global__ void shared_shuffle_const(unsigned int *ordered, unsigned int *shuffled)
{
	__shared__ unsigned int tmp[CELLS];
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	tmp[idx] = ordered[idx];
	__syncthreads();

	shuffled[idx] = tmp[idx];
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

void main_sub()
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (CELLS));

  unsigned int h_puzzle[CELLS] = {0,0,4,3,0,0,2,0,9,
																0,0,5,0,0,9,0,0,1,
																0,7,0,0,6,0,0,4,3,
																0,0,6,0,0,2,0,8,7,
																1,9,0,0,0,7,4,0,0,
																0,5,0,0,8,3,0,0,0,
																6,0,0,0,0,0,1,0,5,
																0,0,3,5,0,8,6,9,0,
																0,4,2,9,1,0,3,0,0};
	unsigned int *h_pinned_puzzle;
  unsigned int *h_solution;

  //pin it
  cudaMallocHost((void **)&h_pinned_puzzle, array_size_in_bytes);
  cudaMallocHost((void **)&h_solution, array_size_in_bytes);

	// Copy it to pinned memory
	memcpy(h_pinned_puzzle, h_puzzle, array_size_in_bytes);

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *d_puzzle;
  unsigned int *d_solution;

  cudaMalloc((void **)&d_puzzle, array_size_in_bytes);
  cudaMalloc((void **)&d_solution, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_puzzle, h_pinned_puzzle, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;

  cudaEvent_t start_time = get_time();

	shared_shuffle_const<<<num_blocks, num_threads>>>(d_puzzle, d_solution);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);

  cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy(h_solution, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

	printf("Puzzle:\n");
	sudoku_print(h_puzzle);

	printf("Solution:\n");
  sudoku_print(h_solution);

	printf("\tSolved in: %fmsn\n", duration);

  /* Free the GPU memory */
  cudaFree(d_puzzle);
  cudaFree(d_solution);

  /* Free the pinned CPU memory */
  cudaFreeHost(h_pinned_puzzle);
	cudaFreeHost(h_solution);
}

/**
 * Prints the correct usage of this file
 * @name is the name of the executable (argv[0])
 */
void print_usage(char *name)
{
  printf("Usage: %s \n", name);
}

/**
 * Entry point for execution. Checks command line arguments
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{
  /* Check the number of arguments, print usage if wrong */
  if(argc != 1) {
    printf("Error: Incorrect number of command line arguments\n");
    print_usage(argv[0]);
    exit(-1);
  }

  printf("\n");

  main_sub();

  return EXIT_SUCCESS;
}
