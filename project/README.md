Class Project
Sarah Helble

Sudoku Solver
-------------
Once finished, this program will be able to solve any sudoku puzzle read in
from a file with efficiency.
Extras that I also want to include:
1. Check that an answer is correct
2. Generate sudoku puzzles of varying complexity
3. Possibly larger puzzles (double-digit sudokus?)

Current Status Summary
----------------------
Currently, this program solves sudoku puzzles contained in a passed input file,
(The file included is the first hundred puzzles from [1]) by calling the solve_by_possibility kernel function until a host function determines that the puzzle has been solved. It only uses the solve_by_possibility kernel. In my
TODO, I identify other approaches that could be alternated to achieve a faster
solution. Sudoku-style output is sent to STDOUT. Call the program without any
command line arguments in order to see other runtime options, such as verbosity
and number of blocks.

Metrics are output to a file called metrics.csv in the format set, block, puzzle,
solution, count, duration. Puzzles are solved in sets of num_blocks, so set is
the set a puzzle belongs to and block is the block it took in that set. Notice
that since a set is being solved simultaneously, count and duration are the time
and duration for the set as a whole, therefore, it ends up being the highest count,
longest duration one's metrics.

Lessons so far
--------------
- sharing across blocks
- debug prints
- first hundred are solvable
- output to csv easier than expected
- makefiles are hard
- code review thanks
- online strategies (naked pair, solve by constraint)
- async cells, sure; async puzzles?
- cudaStreamSynchronize() was necessary inside loop. Hurt time if not
- puzzle per block lessons and speedup
- shared memory when multiple blocks
  - an SM can have up to 8 blocks at the same time
- getting the data back seems to waste a lot of time

### TODO
Shortlist:  1) gather metrics 2) start presentation and report
3) more algorithms 4) gather metrics for those 5) maybe more puzzles?

- host based version of solution
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
