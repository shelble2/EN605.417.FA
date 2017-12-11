/**
 * solver_kernels.cu
 * Sarah Helble
 * 2017-12-07
 *
 * This file contains all of the kernels for the sudoku solver project.
 */

/**
 * Kernel function that solves based on last available. If only one number
 * can fit in a given cell, based on the contents of its row, column, and block;
 * then fill the cell with that value.
 */
__global__ void solve_by_possibility(unsigned int *puzzle, unsigned int *solved)
{
	__shared__ unsigned int shared_puzzle[CELLS];

	const unsigned int local_id = threadIdx.x;
	const unsigned int id = local_id + (blockIdx.x * blockDim.x);

	// copy to shared array
	shared_puzzle[local_id] = puzzle[id];
	__syncthreads();

	// Calculate our row and column
	const unsigned int col = local_id % DIM;
	const unsigned int row = (local_id - col) / DIM;

	// Keep a list of possible values. The values are actually the indices here,
	// a 0 indicates that that index value is no longer a possibility.
	unsigned int possibilities[DIM+1] = {0,1,1,1,1,1,1,1,1,1};

	// Only try to solve if cell is empty
	if(shared_puzzle[local_id] != 0 ) {
		shared_puzzle[local_id]  = shared_puzzle[local_id];
	} else {
		// Rule out all in the same row by changing their value in possibilities
		for(int i = row * DIM; i < ((row*DIM) + DIM); i++) {
			int current = shared_puzzle[i];
			possibilities[current] = 0;
		}

		//Go through all in the same column
		for(int i = 0; i < DIM ; i++) {
			int current = shared_puzzle[(i*DIM)+col];
			possibilities[current] = 0;
		}

		//Go through all in the same block
		// TODO: these could just be integer division instead?
		int s_row = row - (row % B_DIM);
		int s_col = col - (col % B_DIM);
		for(int i = s_row; i < (s_row + B_DIM); i++) {
			for(int j = s_col; j < (s_col + B_DIM); j++) {
				int current = shared_puzzle[(i*DIM)+j];
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

		shared_puzzle[local_id] = candidate;
	}

	__syncthreads();

	solved[id] = shared_puzzle[local_id];
}
