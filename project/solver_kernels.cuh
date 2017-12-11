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
	__shared__ unsigned int tmp[CELLS];

	// Calculate our row and column
	const unsigned int my_cell_id = threadIdx.x;
	const unsigned int col = my_cell_id % DIM;
	const unsigned int row = (my_cell_id - col) / DIM;

	// Keep a list of possible values. The values are actually the indices here,
	// a 0 indicates that that index value is no longer a possibility.
	unsigned int possibilities[DIM+1] = {0,1,1,1,1,1,1,1,1,1};

	tmp[my_cell_id] = puzzle[my_cell_id];

	// Only try to solve if cell is empty
	if(tmp[my_cell_id] != 0 ) {
		tmp[my_cell_id]  = tmp[my_cell_id];
	} else {
		// Rule out all in the same row by changing their value in possibilities
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
		// TODO: these could just be integer division instead?
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

__global__ void solve_mult_by_possibility(unsigned int *puzzle, unsigned int *solved)
{
	__shared__ unsigned int tmp[CELLS * 2];

	const unsigned int diff_due_to_block = (blockIdx.x * blockDim.x);
	// Calculate our row and column
	const unsigned int my_cell_id = threadIdx.x + diff_due_to_block;
	const unsigned int col = my_cell_id % DIM;
	const unsigned int row = (my_cell_id - col) / DIM;

	// Keep a list of possible values. The values are actually the indices here,
	// a 0 indicates that that index value is no longer a possibility.
	unsigned int possibilities[DIM+1] = {0,1,1,1,1,1,1,1,1,1};

	tmp[my_cell_id] = puzzle[my_cell_id];

	// Only try to solve if cell is empty
	if(tmp[my_cell_id] != 0 ) {
		tmp[my_cell_id]  = tmp[my_cell_id];
	} else {
		printf("cell %d. Looking at %d through %d (row)\n", my_cell_id, row*DIM, (row*DIM)+DIM-1);
		// Rule out all in the same row by changing their value in possibilities
		for(int i = row * DIM; i < ((row*DIM) + DIM); i++) {
			int current = tmp[i];
			possibilities[current] = 0;
		}

		//Go through all in the same column
		for(int i = 0; i < DIM ; i++) {
			int current = tmp[i*DIM+col];
			possibilities[current] = 0;
		}


		printf("cell %d. Looking at %d through %d\n", my_cell_id, col+(diff_due_to_block/DIM),
														(DIM-1)*DIM+col+(diff_due_to_block/DIM));
		//Go through all in the same column
		for(int i = 0; i < DIM ; i++) {
			int current = tmp[(i*DIM)+col+(diff_due_to_block/DIM)];
			possibilities[current] = 0;
		}

		//Go through all in the same block
		// TODO: these could just be integer division instead?
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
