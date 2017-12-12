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
#include <unistd.h>

#include "sudoku_utils.cuh"
#include "solver_kernels.cuh"

/**
 * Loops over the kernel until the puzzle is solved or LOOP_LIMIT is reached
 * On success, returns the number of iterations performed, and **solution is
 * set to the end result. Returns -1 on failure.
 * hp_puzzle is the host-pinned puzzle of unsigned ints
 * cells is the number of cells in the puzzle
 * blocks is the number of blocks to use at a time (one puzzle per block)
 * This function handles copying between host and device synchronously
 */
int execute_kernel_loop_sync(unsigned int *hp_puzzles, int cells, int blocks, unsigned int **solutions)
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
 * Loops over the kernel until the puzzle is solved or LOOP_LIMIT is reached
 * On success, returns the number of iterations performed, and **solution is
 * set to the end result. Returns -1 on failure.
 * hp_puzzle is the host-pinned puzzle of unsigned ints
 * cells is the number of cells in the puzzle
 * blocks is the number of blocks to use at a time (one puzzle per block)
 * This function handles copying of data between host and device asynchronously
 */
int execute_kernel_loop_async(unsigned int *hp_puzzles, int cells, int blocks, unsigned int **solutions)
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

	// While the puzzle is not finished, iterate until LOOP_LIMIT is reached
	do {
		/* Copy the CPU memory to the GPU memory */
		cuda_ret = cudaMemcpyAsync(d_puzzles, hp_puzzles, array_size_in_bytes,
									cudaMemcpyHostToDevice, stream);
		if(cuda_ret != cudaSuccess) {
			printf("ERROR memcpy host to device (%d)\n", cuda_ret);
			count = -1;
			goto memcpy_error;
		}

		solve_by_possibility<<<blocks, cells>>>(d_puzzles, d_solutions);

		/* Copy the changed GPU memory back to the CPU */
		cuda_ret = cudaMemcpyAsync(hp_puzzles, d_solutions, array_size_in_bytes,
						cudaMemcpyDeviceToHost, stream);
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
 int solve_puzzles(unsigned int *h_puzzles, int cells, int blocks,
	 				unsigned int **out, int *out_count, float *out_duration)
 {
	 int ret = 0;
	 int array_size_in_bytes = (sizeof(unsigned int) * (cells * blocks));
	 cudaError cuda_ret;
	 *out = NULL;
	 *out_count = 0;
	 *out_duration = 0;

	 //pin it and copy to pinned memory
	 unsigned int *h_pinned_puzzles;
	 unsigned int *solutions;
	 cuda_ret = cudaMallocHost((void **) &h_pinned_puzzles, array_size_in_bytes);
	 if(cuda_ret != cudaSuccess) {
 		printf("Error mallocing pinned host memory\n");
 		return -1;
 	}
 	memcpy(h_pinned_puzzles, h_puzzles, array_size_in_bytes);

 	/* Execute the kernel and keep track of start and end time for duration */
 	float duration = 0;
 	cudaEvent_t start_time = get_time();

 	int count = execute_kernel_loop_sync(h_pinned_puzzles, cells, blocks, &solutions);
 	if(count <= 0) {
 		printf("ERROR: returned %d from execute_kernel_loop\n", count);
 		cudaFreeHost(h_pinned_puzzles);
 		return -1;
 	}

 	cudaEvent_t end_time = get_time();
 	cudaEventSynchronize(end_time);
 	cudaEventElapsedTime(&duration, start_time, end_time);

 	/* Free the pinned CPU memory */
 	cudaFreeHost(h_pinned_puzzles);

	*out = solutions;
	*out_count = count;
	*out_duration = duration;
 	return ret;
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
void solve_mult_from_fp(FILE *input_fp, FILE *metrics_fp, int blocks,
						int verbosity, int *solved, int *unsolvable, int *errors)
{
	char *lines[blocks];
	unsigned int *h_puzzles;
	unsigned int *h_solutions
	char *line = NULL;
	size_t len = 0;

	int tmp_solved = 0;
	int tmp_errors = 0;
	int tmp_unsolvable = 0;
	int ret;

	int set = 0;
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

		// Pass the number of puzzles as the number of blocks
		h_puzzles = host_load_puzzles(lines, i, CELLS);

		if(verbosity == 1) {
	 		sudoku_print_puzzles(h_puzzles, blocks);
	 	}
		int count;
		float duration;
		ret = solve_puzzles(h_puzzles, CELLS, i, &h_solutions, &count, &duration);

		if(verbosity == 1) {
	 		sudoku_print_puzzles(h_puzzles, blocks);
	 		printf("\tSolved in %d increments and %fms\n", count, duration);
	 	}

		//XXX: Could this print to file be a bottleneck?
		if(metrics_fd != NULL) {
			output_mult_metrics_to_file(metrics_fd, blocks, set, h_puzzles,
										h_solutions, count, duration);
		}

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
 * Prints the usage of this Program
 */
 void print_usage(char *name)
 {
	 printf("\nUsage: %s -i [input file] (-b num_blocks) (-v verbosity) (-a)\n", name);
	 printf("\t-i input file is required\n");
	 printf("\t-b is optional and specifies the number of blocks to use, or the number\n");
	 printf("\t\tof puzzles to exec in parallel. Default = 1\n");
	 printf("\t-v specifies verbosity, where -v 0 will suppress most output. Default = 1\n");
	 printf("\t-a is a flag that will result in calling asynchronous functions instead of the synchronous default\n");
 }

/**
 * Entry point for execution. Checks command line arguments
 * then passes execution to subordinate function
 */
int main(int argc, char *argv[])
{
	int verbosity = DEFAULT_VERBOSITY;
	int blocks    = DEFAULT_NUM_BLOCKS;
	int async     = 0;
	char *input_fn = NULL;
	FILE *input_fp = NULL;

	int c;

    while((c = getopt(argc, argv, "v:b:ai:")) != -1) {
      switch(c) {
        case 'v':
			verbosity = atoi(optarg);
          	break;
        case 'b':
			blocks = atoi(optarg);
          	break;
        case 'a':
          	async = 1;
          	break;
		case 'i':
			input_fn = optarg;
			break;
        default:
          printf("Error: unrecognized option: %c\n", c);
		  print_usage(argv[0]);
          exit(-1);
        }
    }

	if(input_fn == NULL) {
		printf("Error: no input file provided\n");
		print_usage(argv[0]);
		return -1;
	}

	input_fp = fopen(input_fn, "r");
	if(input_fp == NULL) {
		printf("Failed to open input file %s\n", input_fn);
		return -1;
	}

	printf("Using %d blocks to solve %d at a time.\n", blocks, blocks);

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
