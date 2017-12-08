NVCC = nvcc

sudoku_solver: sudoku_solver.cu sudoku_utils.h sudoku_utils.cu solver_kernels.cu
	$(NVCC) -o sudoku_solver *.cu
