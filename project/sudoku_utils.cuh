/**
 * sudoku_utils.cu
 * Sarah Helble
 * 2017-12-07
 *
 * This file contains utility functions for the sudoku solver project
 */

#include <stdio.h>
#include <stdlib.h>

#define DIM 9             // Customary sudoku
#define B_DIM 3           // dimension of one sudoku block
#define CELLS DIM * DIM   // 81
#define THREADS_PER_BLOCK DIM // Seems like a nice way to split..

#define LOOP_LIMIT 40  // Just in case we hit one that needs a guess to finish
#define ASCII_TO_INT 48

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
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
void sudoku_print(unsigned int* numbers, int start)
{
	int i;
	int j;
	int block_dim = round(sqrt(DIM));

	printf("\n_________________________________________\n");

	for (i = 0; i < DIM; i++) {

		printf("||");
		for (j = 0; j < DIM; j++) {
			printf(" %u |", numbers[ ( (i*DIM) + j ) + start]);
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
 * Prints the passed array like two sudoku puzzles in ascii art
 * @numbers array to print
 * @blocks the number of puzzles in the array of numbers
 */
void sudoku_print_puzzles(unsigned int* numbers, int blocks)
{
	int i = 0;
	for(i = 0; i < blocks, i++){
		printf("Puzzle %d:\n", i);
		sudoku_print(numbers, i*CELLS);
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
	///TODO: needs to know how many
	for(int i = 0; i < CELLS *2; i++) {
		if(puzzle[i] == 0) {
			return 1;
		}
	}
	return 0;
}

/**
 * Function to load two puzzles into a single array of ints (one before the
 * other). Puzzle1 is the first string to load, puzzle2 is the second
 * cells is the nuber of cells in a single puzzle.
 * Returns the loaded unsigned int puzzles array
 */
unsigned int *load_two_puzzles(char *puzzle1, char *puzzle2, int cells)
{
	int i;
	unsigned int *out = (unsigned int *) malloc(cells * 2 * sizeof(unsigned int));
	for(i = 0; i < cells; i++) {
		out[i] = puzzle1[i] - ASCII_TO_INT;
		out[cells+i] = puzzle2[i] - ASCII_TO_INT;
	}
	return out;
}

/**
 * Function to load the puzzle into the array of ints
 * puzzle is the string to load as a sudoku puzzle
 * cells is the number of cells in the puzzles
 * Returns the loaded unsigned int array puzzle
 */
unsigned int *load_puzzle(char *puzzle, int cells)
{
	int i;
	unsigned int *out = (unsigned int *) malloc(cells *sizeof(unsigned int));
	for (i = 0; i < cells; i++) {
	    out[i] = puzzle[i] - ASCII_TO_INT;
	}
	return out;
}

/**
 * Prints passed parameters to the passed csv file
 * out_fd is the file to write to
 * puzzle is the original puzzle
 * solution is the solved puzzle
 * count is the number of iterations the puzzle took to complete
 * duration is the time the puzzle took to solve
 *
 * outputs to the csv file in the format
 * puzzle,solution,count,duration
 */
void output_metrics_to_file(FILE *out_fd, unsigned int *puzzle,
	unsigned int *solution, int count, float duration, int start)
{
	int i = 0;
	for(i = 0; i < CELLS; i++){
		fprintf(out_fd, "%c", (char)puzzle[i+start] + ASCII_TO_INT);
	}
	fprintf(out_fd, ",");
	for(i = 0; i< CELLS; i++){
		fprintf(out_fd, "%c", (char)solution[i+start] + ASCII_TO_INT);
	}
	fprintf(out_fd, ",%d,%0.3f\n", count, duration);
}

/**
 * Prints multiple puzzle's metrics to the passed FILE
 * blocks is the number of puzzles held in puzzles, puzzles are the originals
 * solutions are the end result, count is the number of iterations it took, and
 * duration is the amount of time in ms
 */
//TODO: count and duration here are time for ALL in block
void output_mult_metrics_to_file(FILE *out_fd, int blocks, unsigned int *puzzle,
	unsigned int *solution, int count, float duration)
{
	int i = 0;
	for (i = 0; i < blocks; i++){
		output_metrics_to_file(out_fd, puzzle, solution, count, duration, i*CELLS);
	}
}
