/**
 * Assignment 08
 * header file for random_square_grid.cu, to make matrix creation for
 * invert_matrix.cu easier.
 *
 * Sarah Helble
 * 22 Oct 2017
 **/

 /**
  * This kernel initializes the states for each input in the array
  * @seed is the seed for the init function
  * @states is an allocated array, where the output of this fuction will be stored
  */
__global__ void init_states(unsigned int seed, curandState_t* states);

/**
 * Given the passed array of states @states, this kernel fills allocated array
 * @numbers with a random int between 0 and MAX_INT
 * @states is the set of states already initialized by CUDA
 * @numbers is an allocated array where this kernel function will put its output
 */
__global__ void fill_grid(curandState_t* states, unsigned int* numbers);

/**
 * Prints the passed array like a sudoku puzzle in ascii art
 * @numbers array to print
 */
 void sudoku_print(unsigned int* numbers);
