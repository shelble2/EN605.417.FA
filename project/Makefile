NVCC = nvcc

sudoku_solver: sudoku_solver.cu sudoku_utils.cuh solver_kernels.cuh
	$(NVCC) -lcudart -o sudoku_solver *.cu
