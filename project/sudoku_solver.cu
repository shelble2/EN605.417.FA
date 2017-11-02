/**
 * sudoku_solver.cu
 * Sarah Helble
 * 2017-11-01
 *
 * In the process of adapting shuffle.cu from Module 5 assigment to work on
 * solving sudoku puzzles in the form of a string of numbers, where 0 indicates
 * an empty cell.
 *
 * Previous program:
 * Shuffles the contents of an array according to a pre-determined pattern
 * Used to test the difference between shared/global and const/global memory
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

/**
 * Function that sets up everything for the kernel function
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 * @global_array is 0 if the array should be global memory, 1 if it should be shared
 * @global_plan is 0 if the plan should be global memory, 1 if it should be constant
 */
void exec()
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (CELLS));
  int i = 0;

  unsigned int *ordered;
  unsigned int *shuffled_result;

  //pin it
  cudaMallocHost((void **)&ordered, array_size_in_bytes);
  cudaMallocHost((void **)&shuffled_result, array_size_in_bytes);

	//TODO: change this to load example sudoku
  for(i = 0; i < CELLS; i++) {
  	ordered[i] = i;
  }

  /* Declare and allocate pointers for GPU based parameters */
  unsigned int *d_ordered;
  unsigned int *d_shuffled_result;

  cudaMalloc((void **)&d_ordered, array_size_in_bytes);
  cudaMalloc((void **)&d_shuffled_result, array_size_in_bytes);

  /* Copy the CPU memory to the GPU memory */
  cudaMemcpy(d_ordered, ordered, array_size_in_bytes, cudaMemcpyHostToDevice);

  /* Designate the number of blocks and threads */
  const unsigned int num_blocks = CELLS/THREADS_PER_BLOCK;
  const unsigned int num_threads = CELLS/num_blocks;

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;

  cudaEvent_t start_time = get_time();

	shared_shuffle_const<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);

  cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( shuffled_result, d_shuffled_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("\tDuration: %fmsn\n", duration);
  sudoku_print(ordered, shuffled_result);

  /* Free the GPU memory */
  cudaFree(d_ordered);
  cudaFree(d_shuffled_result);

  /* Free the pinned CPU memory */
  cudaFreeHost(ordered);
  cudaFreeHost(shuffled_result);
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

  exec();

  return EXIT_SUCCESS;
}
