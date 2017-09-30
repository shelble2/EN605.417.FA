/**
 * Assignment 05 Program - shuffle.cu
 * Sarah Helble
 * 9/29/17
 *
 * Shuffles the contents of an array according to a pre-determined pattern
 * Used to test the difference between shared/global and const/global memory
 * 
 * Usage ./aout
 *
 */

#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 512
#define THREADS_PER_BLOCK 16

#define PLAN_DEPTH 10

__constant__ int const_plan[PLAN_DEPTH];
__device__ int gmem_plan[PLAN_DEPTH];

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
 * Kernel function that shuffles the values in @ordered and puts the
 * output in @shuffled
 * With the Plan given below, Example:
 *    [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] Becomes
 *    [ 4, 2, 0, 7, 1, 3, 6, 9, 5, 8 ]
 */
__global__ void shuffle_const(unsigned int *ordered, unsigned int *shuffled)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  unsigned int instruction_num = idx % PLAN_DEPTH;
  int instruction = const_plan[instruction_num];

  shuffled[idx + instruction] = ordered[idx];
}

__global__ void shuffle_gmem(unsigned int *ordered, unsigned int *shuffled)
{
  /* Calculate the current index */
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  unsigned int instruction_num = idx % PLAN_DEPTH;
  int instruction = gmem_plan[instruction_num];

  shuffled[idx + instruction] = ordered[idx];
}

__global__ void shared_shuffle_const(unsigned int *ordered, unsigned int *shuffled)
{
	__shared__ unsigned int tmp[NUM_ELEMENTS];
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int instruction_num = idx % PLAN_DEPTH;
  int instruction = const_plan[instruction_num];

	tmp[idx] = ordered[idx];
	__syncthreads();

	shuffled[idx+instruction] = tmp[idx];
}

__global__ void shared_shuffle_gmem(unsigned int *ordered, unsigned int *shuffled)
{
	__shared__ unsigned int tmp[NUM_ELEMENTS];
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int instruction_num = idx % PLAN_DEPTH;
  int instruction = gmem_plan[instruction_num];

	tmp[idx] = ordered[idx];
	__syncthreads();

	shuffled[idx+instruction] = tmp[idx];
}

/**
 * One fuction to handle the printing of results.
 * @ordered is the original array
 * @shuffled is the result
 */
void print_results(unsigned int *ordered, unsigned int *shuffled)
{
  int i = 0;

  printf("\n");
  for(i = 0; i < NUM_ELEMENTS; i++) {
    printf("Original value at index [%d]: %d, shuffled: %d\n", i, ordered[i], shuffled[i]);
  }
  printf("\n");
}

/**
 * Function that sets up everything for the kernel function
 *
 * @array_size size of array (total number of threads)
 * @threads_per_block number of threads to put in each block
 * @global_array is 0 if the array should be global memory, 1 if it should be shared
 * @global_plan is 0 if the plan should be global memory, 1 if it should be constant
 */
void exec_shuffle(int global_array, int global_plan)
{
  /* Calculate the size of the array */
  int array_size_in_bytes = (sizeof(unsigned int) * (NUM_ELEMENTS));
  int i = 0;

  unsigned int *ordered;
  unsigned int *shuffled_result;
	int plan[PLAN_DEPTH] = {2, 3, -1, 2, -4, 3, 0, -4, 1, -2};

  //pin it
  cudaMallocHost((void **)&ordered, array_size_in_bytes);
  cudaMallocHost((void **)&shuffled_result, array_size_in_bytes);

  /* Read characters from the input and key files into the text and key arrays respectively */
  for(i = 0; i < NUM_ELEMENTS; i++) {
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
  const unsigned int num_blocks = NUM_ELEMENTS/THREADS_PER_BLOCK;
  const unsigned int num_threads = NUM_ELEMENTS/num_blocks;

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;

	cudaEvent_t start_time = get_time();

	if(global_plan == 0 && global_array == 0) {
		// Everything global
		printf("Global Array, Global Plan:\n");
		cudaMemcpyToSymbol(gmem_plan, plan, PLAN_DEPTH * sizeof(int));
		shuffle_gmem<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);
	} else if(global_plan == 0 && global_array == 1) {
		// Plan still global, but array shared
		printf("Shared Array, Global Plan:\n");
		cudaMemcpyToSymbol(gmem_plan, plan, PLAN_DEPTH * sizeof(int));
		shared_shuffle_gmem<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);
	} else if(global_plan == 1 && global_array == 0) {
		// Plan Constant Mem, But Array back to global
		printf("Global Array, Constant Plan:\n");
		cudaMemcpyToSymbol(const_plan, plan, PLAN_DEPTH * sizeof(int));
		shuffle_const<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);
	} else if(global_plan == 1 && global_array == 1) {
		// Plan Constant Mem, But Array back to global
		printf("Shared Array, Constant Plan:\n");
		cudaMemcpyToSymbol(const_plan, plan, PLAN_DEPTH * sizeof(int));
		shared_shuffle_const<<<num_blocks, num_threads>>>(d_ordered, d_shuffled_result);
	}

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);

	cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy( shuffled_result, d_shuffled_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("\tDuration: %fmsn\n", duration);
  print_results(ordered, shuffled_result);

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

	/* Do the shuffle with all global memory */
  exec_shuffle(0, 0);
	printf("-----------------------------------------------------------------\n");

	/* Do the shuffle with shared memory for array, global plan */
	exec_shuffle(1, 0);
	printf("-----------------------------------------------------------------\n");

	/* Do the shuffle with global memory for array, constants for plan */
	exec_shuffle(0, 1);
	printf("-----------------------------------------------------------------\n");

	/* Do the shuffle with shared memory for array, constants for plan */
	exec_shuffle(1, 1);

  return EXIT_SUCCESS;
}
