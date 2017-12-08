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
 * Solves the passed puzzle
 */
int solve_puzzle(unsigned int *h_puzzle, int cells, FILE *metrics_fd)
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

	if(metrics_fd != NULL) {
		output_metrics_to_file(metrics_fd, h_puzzle, h_pinned_puzzle, count, duration);
	}

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

	// Solve each puzzle in the input file
	char *line = NULL;
	size_t len = 0;
	while(getline(&line, &len, input_fp) != -1) {
		unsigned int *h_puzzle = load_puzzle(line, CELLS);
		solve_puzzle(h_puzzle, CELLS, metrics_fp);
		//TODO: Maybe would be better if it returned the result and
		// it's saved in file.
	}

	fclose(input_fp);
	fclose(metrics_fp);

	return EXIT_SUCCESS;
}
