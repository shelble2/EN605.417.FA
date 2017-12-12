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
 * blocks is the number of blocks to use at a time (one puzzle per block)
 */
int execute_kernel_loop(unsigned int *hp_puzzles, int cells, int blocks, unsigned int **solutions)
{
	int count = 0;
	int array_size_in_bytes = (sizeof(unsigned int)*(cells*blocks));
	cudaError cuda_ret;
	*solutions = NULL;

	unsigned int *d_puzzles;
	unsigned int *d_solutions;
	cuda_ret = cudaMalloc((void **)&d_puzzles, array_size_in_bytes);
	if(cuda_ret != cudaSuccess) {
		printf("ERROR in cudaMalloc for d_puzzle\n");
		count = -1;
		goto malloc_puzzle_error;
	}
	cuda_ret = cudaMalloc((void **)&d_solutions, array_size_in_bytes);
	if(cuda_ret != cudaSuccess) {
		printf("ERROR in cudaMalloc for d_solution\n");
		count = -1;
		goto malloc_solution_error;
	}

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	//TODO: make it async again
	// While the puzzle is not finished, iterate until LOOP_LIMIT is reached
	do {
		/* Copy the CPU memory to the GPU memory */
		cuda_ret = cudaMemcpy(d_puzzles, hp_puzzles, array_size_in_bytes,
									cudaMemcpyHostToDevice);
		if(cuda_ret != cudaSuccess) {
			printf("ERROR memcpy host to device (%d)\n", cuda_ret);
			count = -1;
			goto memcpy_error;
		}

		solve_by_possibility<<<blocks, cells>>>(d_puzzles, d_solutions);

		/* Copy the changed GPU memory back to the CPU */
		cuda_ret = cudaMemcpy(hp_puzzles, d_solutions, array_size_in_bytes,
						cudaMemcpyDeviceToHost);
		if(cuda_ret != cudaSuccess) {
			printf("ERROR memcpy device to host (%d)\n", cuda_ret);
			count = -1;
			goto memcpy_error;
		}

		cudaStreamSynchronize(stream);
		count = count + 1;
		//TODO: if we could check each puzzle individually, we could swap one out
		// for another instead of having to wait for least common denominator
	} while ((check_if_done(hp_puzzles, blocks) == 1) && (count <= LOOP_LIMIT));

	if(count == LOOP_LIMIT) {
		printf("[ WARNING ] Could not find a solution within max allowable (%d) iterations.\n", LOOP_LIMIT);
	}

	*solutions = hp_puzzles;

memcpy_error:
	cudaFree(d_solutions);
malloc_solution_error:
	cudaFree(d_puzzles);
malloc_puzzle_error:
	return count;
}

/**
 * Solves the passed puzzles (one per block)
 * h_puzzles is the host array of ints that form the puzzles,
 * cells is the number of cells in each puzzle
 * blocks is the number of puzzles in the array
 * metrics_fd is an open file descriptor to the output file
 * verbosity is a flag for extra prints. If == 1, will print every puzzle and
 * solution to STDOUT. Otherwise, will just print batch metrics. Either way,
 * metrics for each specific puzzle can be found in output file
 */
 int solve_puzzles(unsigned int *h_puzzles, int cells, int blocks, FILE *metrics_fd, int verbosity)
 {
	 int ret = 0;
	 int array_size_in_bytes = (sizeof(unsigned int) * (cells * blocks));
	 cudaError cuda_ret;

	 //pin it and copy to pinned memory
	 unsigned int *h_pinned_puzzles;
	 unsigned int *solutions;
	 cuda_ret = cudaMallocHost((void **) &h_pinned_puzzles, array_size_in_bytes);
	 if(cuda_ret != cudaSuccess) {
 		printf("Error mallocing pinned host memory\n");
 		return -1;
 	}
 	memcpy(h_pinned_puzzles, h_puzzles, array_size_in_bytes);

 	if(verbosity == 1) {
 		sudoku_print_puzzles(h_puzzles, blocks);
 	}

 	/* Execute the kernel and keep track of start and end time for duration */
 	float duration = 0;
 	cudaEvent_t start_time = get_time();

 	int count = execute_kernel_loop(h_pinned_puzzles, cells, blocks, &solutions);
 	if(count <= 0) {
 		printf("ERROR: returned %d from execute_kernel_loop\n", count);
 		cudaFreeHost(h_pinned_puzzles);
 		return -1;
 	}

 	cudaEvent_t end_time = get_time();
 	cudaEventSynchronize(end_time);
 	cudaEventElapsedTime(&duration, start_time, end_time);

 	if(verbosity == 1) {
 		sudoku_print_puzzles(h_pinned_puzzles, blocks);
 		printf("\tSolved in %d increments and %fms\n", count, duration);
 	}

 	//XXX: Could this print to file be a bottleneck?
 	if(metrics_fd != NULL) {
 		output_mult_metrics_to_file(metrics_fd, blocks, h_puzzles, h_pinned_puzzles, count, duration);
 	}

 	/* Free the pinned CPU memory */
 	cudaFreeHost(h_pinned_puzzles);
 	return ret;
 }

/**
 * Find the best available device for our use case and set it
 * Right now, this just picks the one with the highest number of
 * multiprocessors.
 */
void find_and_select_device()
{
	printf("----------------------------------------------\n");
	printf("Finding the best device for the job\n");
	int num_devices;
	int device = 0;
	int max_mp = 0;
	int i;

	// Figure out how many devices there are
	cudaGetDeviceCount(&num_devices);
	printf("%d possible devices\n", num_devices);

	printf("Selecting the one with the highest number of multiprocessors\n");
	for(i = 0; i < num_devices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		if(prop.multiProcessorCount > max_mp) {
			max_mp = prop.multiProcessorCount;
			device = i;
		}
		printf("Device %d has : \n\t%d multiprocessors\n\t%d warp size\n",
				i, prop.multiProcessorCount, prop.warpSize);
	}

	printf("Selected device %d\n", device);
	cudaSetDevice(device);
	printf("----------------------------------------------\n");
}

/**
 * Loads the lines from the open file descriptor one by one and solves them
 * input_fp is the open file descriptor to read from
 * metrics_fp is an open file descriptor to write metrics to
 * Does not return a value, but sets solved to the number of puzzles
 * successfully finished, unsolved to the number that could not be Solved within
 * the LOOP_LIMIT, and sets error to the number of puzzles that returned with
 * error
 */
//TODO: would it be better to have each puzzle as array of int in array of ints?
void solve_mult_from_fp(FILE *input_fp, FILE *metrics_fp, int blocks,
						int verbosity, int *solved, int *unsolvable, int *errors)
{
	char *lines[blocks];
	unsigned int *h_puzzles;
	char *line = NULL;
	size_t len = 0;

	int tmp_solved = 0;
	int tmp_errors = 0;
	int tmp_unsolvable = 0;
	int ret;

	// while there's still at least one left
	while(getline(&line, &len, input_fp) != -1) {
		lines[0] = strdup(line);

		// Fill in the rest of the space
		int i;
		for(i = 1; i < blocks; i++) {
			if(getline(&line, &len, input_fp) == -1) {
				break;
			}
			lines[i] = strdup(line);
		}

		h_puzzles = load_puzzles(lines, i, CELLS);

		ret = solve_puzzles(h_puzzles, CELLS, i, metrics_fp, verbosity);

		//TODO: Need a better way to handle errors and counts, as this is per set, not block
		if(ret == -1) {
			tmp_errors = tmp_errors + 1;
		} else if(ret == LOOP_LIMIT) {
			tmp_unsolvable = tmp_unsolvable + 1;
		} else {
			tmp_solved = tmp_solved + 1;
		}
	}

	*solved = tmp_solved;
	*unsolvable = tmp_unsolvable;
	*errors = tmp_errors;
}

/**
 * Entry point for execution. Checks command line arguments
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{
	int verbosity = 1;
	if(argc != 3 && argc != 4) {
		printf("Error: Incorrect number of command line arguments\n");
		printf("Usage: %s [input_file] [num_blocks] (v=0)\n", argv[0]);
		exit(-1);
	}
	printf("\n");

	char *input_fn = argv[1];
	FILE *input_fp = fopen(input_fn, "r");
	if(input_fp == NULL) {
		printf("Failed to open input file %s\n", input_fn);
		return -1;
	}

	int blocks = atoi(argv[2]);
	printf("Using %d blocks to solve %d at a time\n", blocks, blocks);


	// TODO: this would be prettier if switched to optparse
	if((argc == 4) && (strcmp(argv[3], "v=0") == 0)) {
		verbosity = 0;
	}

	//TODO: make this a command line option instead of Hardcoded
	char *metrics_fn = "metrics.csv";
	FILE *metrics_fp = fopen(metrics_fn, "w");
	if(metrics_fp == NULL) {
		printf("Failed to open metrics file for writing\n");
		fclose(input_fp);
		return -1;
	}

	find_and_select_device();

	/* Keep track of total duration */
	float duration = 0;
	cudaEvent_t start_time = get_time();

	int solved;
	int unsolvable;
	int errors;

	solve_mult_from_fp(input_fp, metrics_fp, blocks, verbosity,
						&solved, &unsolvable, &errors);

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);
	cudaEventElapsedTime(&duration, start_time, end_time);

	printf("\nFrom a dataset of %d puzzles,\n", solved + unsolvable + errors);
	printf("Solved %d, partially solved %d, and encountered %d errors in %0.3fms\n\n", solved, unsolvable, errors, duration);
	printf("Individual puzzle data output to %s\n", metrics_fn);

	fclose(input_fp);
	fclose(metrics_fp);

	return EXIT_SUCCESS;
}
