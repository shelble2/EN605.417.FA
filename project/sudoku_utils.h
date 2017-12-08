/**
 * sudoku_utils.h
 * Sarah Helble
 * 2017-12-07
 *
 * This is the header file for the sudoku_utils.c file, which contains utility
 * functions for the sudoku solver project.
 */

#define DIM 9             // Customary sudoku
#define B_DIM 3           // dimension of one sudoku block
#define CELLS DIM * DIM   // 81
#define THREADS_PER_BLOCK DIM // Seems like a nice way to split..

#define LOOP_LIMIT 20  // Just in case we hit one that needs a guess to finish
#define ASCII_TO_INT 48

/**
 * Returns the current time
 */
__host__ cudaEvent_t get_time(void);

/**
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
void sudoku_print(unsigned int* numbers);


/**
 * Goes through the puzzle and makes sure each cell block has
 * been filled.
 * puzzle- sudoku puzzle array
 * Returns 0 if done, 1 if not
 */
int check_if_done(unsigned int *puzzle);

/**
 * Function to load the puzzle into the array of ints
 * Hardcoded to this puzzle for now
 */
unsigned int *load_puzzle(char *puzzle, int cells);

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
	unsigned int *solution, int count, float duration);
