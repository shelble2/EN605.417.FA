/**
 * solver_kernels.cuh
 * Sarah Helble
 * 2017-12-07
 *
 * Header file for the sudoku_kernels.cu file, which houses all of the kernels for the
 * sudoku solver project
 */

/**
 * Kernel function that solves based on last available. If only one number
 * can fit in a given cell, based on the contents of its row, column, and block;
 * then fill the cell with that value.
 */
__global__ void solve_by_possibility(unsigned int *ordered, unsigned int *solved);
