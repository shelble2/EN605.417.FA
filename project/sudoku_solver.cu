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

#include <stdio.h>
#include <stdlib.h>

#define DIM 9             // Customary sudoku
#define B_DIM 3           // dimension of one sudoku block
#define CELLS DIM * DIM   // 81
#define THREADS_PER_BLOCK DIM // Seems like a nice way to split..

#define LOOP_LIMIT 20  // Just in case we hit one that needs a guess to finish

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

/**
 * Goes through the puzzle and makes sure each cell block has
 * been filled.
 * puzzle- sudoku puzzle array
 * Returns 0 if done, 1 if not
 */
int check_if_done(unsigned int *puzzle)
{
	for(int i = 0; i < CELLS; i++) {
		if(puzzle[i] == 0) {
			return 1;
		}
	}
	return 0;
}

/**
 * Function to load the puzzle into the array of ints
 * Hardcoded to this puzzle for now
 */
unsigned int *load_puzzle(int cells)
{
	int i;
	unsigned int hardcoded_sudoku[CELLS] = {0,0,4,3,0,0,2,0,9,
		0,0,5,0,0,9,0,0,1,
		0,7,0,0,6,0,0,4,3,
		0,0,6,0,0,2,0,8,7,
		1,9,0,0,0,7,4,0,0,
		0,5,0,0,8,3,0,0,0,
		6,0,0,0,0,0,1,0,5,
		0,0,3,5,0,8,6,9,0,
		0,4,2,9,1,0,3,0,0};
	unsigned int *out = (unsigned int *) malloc(cells *sizeof(unsigned int));
	for (i = 0; i < cells; i++) {
	    out[i] = hardcoded_sudoku[i];
	}
	return out;
}

/**
 * Solves the passed puzzle
 */
int solve_puzzle(unsigned int *h_puzzle, int cells)
{
	int ret = 0;
	int array_size_in_bytes = (sizeof(unsigned int) * (cells));

	//pin it and copy to pinned memory
	unsigned int *h_pinned_puzzle;
	cudaMallocHost((void **)&h_pinned_puzzle, array_size_in_bytes);
	memcpy(h_pinned_puzzle, h_puzzle, array_size_in_bytes);

	/* Declare and allocate pointers for GPU based parameters */
	unsigned int *d_puzzle;
	unsigned int *d_solution;
	cudaMalloc((void **)&d_puzzle, array_size_in_bytes);
	cudaMalloc((void **)&d_solution, array_size_in_bytes);

	printf("Puzzle:\n");
	sudoku_print(h_puzzle);

	/* Execute the kernel and keep track of start and end time for duration */
	float duration = 0;
	cudaEvent_t start_time = get_time();

	int count = 0;
	do {
		/* Copy the CPU memory to the GPU memory */
		cudaMemcpy(d_puzzle, h_pinned_puzzle, array_size_in_bytes, cudaMemcpyHostToDevice);

		solve_by_possibility<<<1, CELLS>>>(d_puzzle, d_solution);

		/* Copy the changed GPU memory back to the CPU */
		cudaMemcpy(h_pinned_puzzle, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);

		count = count + 1;
	} while ((check_if_done(h_pinned_puzzle) == 1) && (count <= LOOP_LIMIT));

	if(count == LOOP_LIMIT) {
		ret = -1;
		printf("Could not find a solution within %d iterations. Here's as far as we got..\n", LOOP_LIMIT);
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	cudaEventElapsedTime(&duration, start_time, end_time);

	printf("Solution:\n");
	sudoku_print(h_pinned_puzzle);

	printf("\tSolved in %d increments and %fms\n", count, duration);

	/* Free the GPU memory */
	cudaFree(d_puzzle);
	cudaFree(d_solution);

	/* Free the pinned CPU memory */
	cudaFreeHost(h_pinned_puzzle);

	return ret;
}

/**
 * Entry point for execution. Checks command line arguments
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{
	if(argc != 2) {
		printf("Error: Incorrect number of command line arguments\n");
		printf("Usage: %s [input_file]\n", argv[0]);
		exit(-1);
	}
	printf("\n");

	char *input_fn = argv[1];
	FILE *input_fp = open(input_fn, "r");
	if(input_fp == NULL) {
		printf("Failed to open input file %s\n", input_fn);
		return -1;
	}

	// Solve each puzzle in the input file
	char *line = NULL;
	size_t len = 0;
	ssize_t read = 0;
	while(read = getline(&line, &len, input_fp) != -1) {
		unsigned int *h_puzzle = load_puzzle(CELLS);
		ret = solve_puzzle(h_puzzle, CELLS);
		//TODO: Maybe would be better if it returned the result and
		// it's saved in file.
	}

	return EXIT_SUCCESS;
}
