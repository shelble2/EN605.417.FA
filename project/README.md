Sudoku Solver
=============

Author: Sarah Helble
This program is able to solve a large number of sudoku puzzles in a very short
amount of time.

Quick Start
-----------

Run

`$ make`

That will result in the creation of `sudoku_solver` executable.
View the usage of the program with

`$ ./sudoku_solver`

This directory contains example puzzles in `puzzles.txt` and `100K_puzzles.txt`.
I recommend suppressing output by passing `-v 0` to the program. For example

`$ ./sudoku_solver -i puzzles.txt -v 0`

If you want to change the number of blocks used, use the `-b [num]` flag, as so

`$ ./sudoku_solver -i puzzles.txt -v 0 -b 100`

Current Status Summary
----------------------
Currently, this program solves sudoku puzzles contained in a passed input file,
(this directory contains txt files with the first hundred and the first 100K
puzzles from [1]) by calling the solve_by_possibility kernel function until a host function determines that the puzzle has been solved. There is only the one kernel.
In Future Work, I identify other approaches that could be alternated to achieve a
faster solution. Sudoku-style output is sent to STDOUT. Call the program without
any command line arguments in order to see other runtime options, such as
verbosity and number of blocks.

Metrics are output to a file called metrics.csv in the format set, block, puzzle,
solution, count, duration. Puzzles are solved in sets of num_blocks, so set is
the set a puzzle belongs to and block is the block it took in that set. Notice
that since a set is being solved simultaneously, count and duration are the time
and duration for the set as a whole, therefore, it ends up being the highest count,
longest duration one's metrics.

### Future Work
- host based version of solution to compare
- kernel version of host_load_puzzles
- different algorithms
- host-based versions for comparison
- another kernel that solves by dimension (check a row for last values to fill)
- switch to solving by constraint
- 'naked pairs' - if two cells both are between a pair, you can't tell which
	has which, but you can tell that they contain both. So can eliminate from
	other dependencies.

### Optional Extras
- larger puzzles going > single digits
- hard sudokus sometimes require guessing between two options for a cell - how to handle?
- generate a sudoku puzzle
  - started in module 8 work. Generates a grid of the appropriate size filled with
    ints in the right range, but it doesn't follow the rules of sudoku
- CI
- check that the answer is correct. For now, just checking against answer in db

[1] https://www.kaggle.com/bryanpark/sudoku/kernels
