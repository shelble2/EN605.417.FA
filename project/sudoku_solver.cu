/**
 * sudoku_solver.cu
 * Sarah Helble
 * 2017-11-14
 *
 * In the process of adapting shuffle.cu from Module 5 assigment to work on
 * solving sudoku puzzles in the form of a string of numbers, where 0 indicates
 * an empty cell.
 *
 * Should compile with `$ nvcc sudoku_solver.cu -o sudoku_solver` and run with
 * `$ ./sudoku_solver`
 */
/*
Lessons so far
- sharing accross blocks
- debug prints
*/

#include <stdio.h>
#include <stdlib.h>

#define DIM 9             // Customary sudoku
#define B_DIM 3           // dimension of one sudoku block
#define CELLS DIM * DIM   // 81
#define THREADS_PER_BLOCK DIM // Seems like a nice way to split..

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
 * Kernel function that solves based on last available. If only one number
 * can fit in a given cell, based on the contents of its row, column, and block;
 * then fill the cell with that value.
 */
__global__ void solve_by_possibility(unsigned int *ordered, unsigned int *solved)
{
	__shared__ unsigned int tmp[CELLS];

	const unsigned int my_cell_id = threadIdx.x;
	const unsigned int col = my_cell_id % DIM;
	const unsigned int row = (my_cell_id - col) / DIM;

	unsigned int possibilities[DIM+1] = {0,1,1,1,1,1,1,1,1,1};

	tmp[my_cell_id] = ordered[my_cell_id];

	// Only try to solve if cell is empty
	if(tmp[my_cell_id] != 0 ) {
		tmp[my_cell_id]  = tmp[my_cell_id];
	} else {
		// Go through all in the same row
		for(int i = row * DIM; i < ((row*DIM) + DIM); i++) {
			int current = tmp[i];
			possibilities[current] = 0;
		}

		//Go through all in the same column
		for(int i = 0; i < DIM ; i++) {
			int current = tmp[i*DIM+col];
			possibilities[current] = 0;
		}

		//Go through all in the same block
		int s_row = row - (row % B_DIM);
		int s_col = col - (col % B_DIM);
		for(int i = s_row; i < (s_row + B_DIM); i++) {
			for(int j = s_col; j < (s_col + B_DIM); j++) {
				int current = tmp[(i*DIM)+j];
				possibilities[current] = 0;
			}
		}

		int candidate = 0;

		// If only one possibility is left, use it
		for(int i = 0; i < DIM+1; i++) {
			if(possibilities[i] == 1) {
				if (candidate == 0) {
					candidate = i;
				} else {
					candidate = 0;
					break;
				}
			}
		}

		tmp[my_cell_id] = candidate;
	}

	__syncthreads();

	solved[my_cell_id] = tmp[my_cell_id];
}

/**
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
 void sudoku_print(unsigned int* numbers)
{
  int i;
  int j;
  int block_dim = round(sqrt(DIM));

  printf("\n_________________________________________\n");

  for (i = 0; i < DIM; i++) {

    printf("||");
    for (j = 0; j < DIM; j++) {
      printf(" %u |", numbers[ ( (i*DIM) + j ) ]);
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

int check_if_done()

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

  printf("Puzzle:\n");
  sudoku_print(h_puzzle);

  /* Execute the kernel and keep track of start and end time for duration */
  float duration = 0;

  cudaEvent_t start_time = get_time();

  //SCH: used to be num_blocks, num_threads, but think all has to be on same block to share
  solve_by_possibility<<<1, CELLS>>>(d_puzzle, d_solution);

  cudaEvent_t end_time = get_time();
  cudaEventSynchronize(end_time);

  cudaEventElapsedTime(&duration, start_time, end_time);

  /* Copy the changed GPU memory back to the CPU */
  cudaMemcpy(h_solution, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Increment 1:\n");
  sudoku_print(h_solution);

  cudaMemcpy(d_puzzle, h_solution, array_size_in_bytes, cudaMemcpyHostToDevice);

  solve_by_possibility<<<1,CELLS>>>(d_puzzle, d_solution);

  cudaMemcpy(h_solution, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Increment 2:\n");
  sudoku_print(h_solution);

  cudaMemcpy(d_puzzle, h_solution, array_size_in_bytes, cudaMemcpyHostToDevice);

  solve_by_possibility<<<1,CELLS>>>(d_puzzle, d_solution);

  cudaMemcpy(h_solution, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

  printf("Increment 3:\n");
  sudoku_print(h_solution);

  cudaMemcpy(d_puzzle, h_solution, array_size_in_bytes, cudaMemcpyHostToDevice);

  solve_by_possibility<<<1,CELLS>>>(d_puzzle, d_solution);

  cudaMemcpy(h_solution, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

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
