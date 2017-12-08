/**
 * sudoku_solver.cu
 * Sarah Helble
 * 2017-12-07
 *
 * Top-level file for the sudoku solver project.
 * This includes the main function but calls out to the sudoku_utils and
 * solver_kernels for most operations.
 */

#include <stdio.h>
#include <stdlib.h>

#include "sudoku_utils.cuh"
#include "solver_kernels.cuh"

/**
 * Loops over the kernel until the puzzle is solved or LOOP_LIMIT is reached
 * On success, returns the number of iterations performed, and **solution is
 * set to the end result. Returns -1 on failure.
 * hp_puzzle is the host-pinned puzzle of unsigned ints
 * cells is the number of cells in the puzzle
 */
int execute_kernel_loop(unsigned int *hp_puzzle, int cells, unsigned int **solution)
{
	int count = 0;
	int array_size_in_bytes = (sizeof(unsigned int) * (cells));
	cudaError cuda_ret;
	*solution = NULL;

	/* Declare and allocate pointers for GPU based parameters */
	unsigned int *d_puzzle;
	unsigned int *d_solution;
	cuda_ret = cudaMalloc((void **)&d_puzzle, array_size_in_bytes);
	if(cuda_ret != cudaSuccess) {
		printf("ERROR in cudaMalloc for d_puzzle\n");
		count = -1;
		goto malloc_puzzle_error;
	}
	cuda_ret = cudaMalloc((void **)&d_solution, array_size_in_bytes);
	if(cuda_ret != cudaSuccess) {
		printf("ERROR in cudaMalloc for d_solution\n");
		count = -1;
		goto malloc_solution_error;
	}

	// While the puzzle is not finished, iterate until LOOP_LIMIT is reached
	do {
		/* Copy the CPU memory to the GPU memory */
		cuda_ret = cudaMemcpy(d_puzzle, hp_puzzle, array_size_in_bytes, cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			printf("ERROR memcpy host to device\n");
			count = -1;
			goto memcpy_error;
		}

		solve_by_possibility<<<1, cells>>>(d_puzzle, d_solution);

		/* Copy the changed GPU memory back to the CPU */
		cudaMemcpy(hp_puzzle, d_solution, array_size_in_bytes, cudaMemcpyDeviceToHost);
		if(cuda_ret != cudaSuccess) {
			printf("ERROR memcpy host to device\n");
			count = -1;
			goto memcpy_error;
		}

		count = count + 1;
	} while ((check_if_done(hp_puzzle) == 1) && (count <= LOOP_LIMIT));

	if(count == LOOP_LIMIT) {
		printf("[ WARNING ] Could not find a solution within max allowable (%d) iterations.\n", LOOP_LIMIT);
	}

	*solution = hp_puzzle;

memcpy_error:
	cudaFree(d_solution);
malloc_solution_error:
	cudaFree(d_puzzle);
malloc_puzzle_error:
	return count;
}

/**
 * Solves the passed puzzle
 * h_puzzle is the host array of ints that form the puzzle,
 * cells is the number of cells in the puzzle
 * metrics_fd is an open file descriptor to the output file
 */
int solve_puzzle(unsigned int *h_puzzle, int cells, FILE *metrics_fd)
{
	int ret = 0;
	int array_size_in_bytes = (sizeof(unsigned int) * (cells));
	cudaError cuda_ret;

	//pin it and copy to pinned memory
	unsigned int *h_pinned_puzzle;
	unsigned int *solution;
	cuda_ret = cudaMallocHost((void **)&h_pinned_puzzle, array_size_in_bytes);
	if(cuda_ret != cudaSuccess) {
		printf("Error mallocing pinned host memory\n");
		return -1;
	}
	memcpy(h_pinned_puzzle, h_puzzle, array_size_in_bytes);

	printf("Puzzle:\n");
	sudoku_print(h_puzzle);

	/* Execute the kernel and keep track of start and end time for duration */
	float duration = 0;
	cudaEvent_t start_time = get_time();

	int count = execute_kernel_loop(h_pinned_puzzle, cells, &solution);
	if(count <= 0) {
		printf("ERROR: returned %d from execute_kernel_loop\n", count);
		cudaFreeHost(h_pinned_puzzle);
		return -1;
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

	printf("Solution:\n");
	sudoku_print(h_pinned_puzzle);

	printf("\tSolved in %d increments and %fms\n", count, duration);

	if(metrics_fd != NULL) {
		output_metrics_to_file(metrics_fd, h_puzzle, h_pinned_puzzle, count, duration);
	}

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
	int ret = 0;
	if(argc != 2) {
		printf("Error: Incorrect number of command line arguments\n");
		printf("Usage: %s [input_file]\n", argv[0]);
		exit(-1);
	}
	printf("\n");

	char *input_fn = argv[1];
	FILE *input_fp = fopen(input_fn, "r");
	if(input_fp == NULL) {
		printf("Failed to open input file %s\n", input_fn);
		return -1;
	}

	//TODO: make this a command line option instead of Hardcoded
	char *metrics_fn = "metrics.csv";
	FILE *metrics_fp = fopen(metrics_fn, "w");
	if(metrics_fp == NULL) {
		printf("Failed to open metrics file for writing\n");
		fclose(input_fp);
		return -1;
	}

	/* Keep track of total duration */
	float duration = 0;
	cudaEvent_t start_time = get_time();

	// Solve each puzzle in the input file
	char *line = NULL;
	size_t len = 0;
	int solved = 0;
	int errors = 0;
	int unsolvable = 0;
	while(getline(&line, &len, input_fp) != -1) {
		unsigned int *h_puzzle = load_puzzle(line, CELLS);
		ret = solve_puzzle(h_puzzle, CELLS, metrics_fp);

		// Keep track of the statuses coming out
		if(ret == -1) {
			errors = errors + 1;
		} else if(ret == LOOP_LIMIT) {
			unsolvable = unsolvable + 1;
		} else {
			solved = solved + 1;
		}
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

	printf("\nFrom a dataset of %d puzzles,\n", solved + unsolvable + errors);
	printf("Solved %d, partially solved %d, and encountered %d errors in %0.3fms\n\n", solved, unsolvable, errors, duration);

	fclose(input_fp);
	fclose(metrics_fp);

	return EXIT_SUCCESS;
}
