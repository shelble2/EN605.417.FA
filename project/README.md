Class Project
Sarah Helble
14 Nov 2017

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
solution.

Lessons so far
--------------
- sharing across blocks
- debug prints
- first hundred are solvable
- better algorithms

### TODO
- different algorithms
  - another kernel that solves by dimension (check a row for last values to fill)
  - switch to solving by constraint
- Perfect solving a lot of sudoku puzzles at once
  - how does this change the blocks/threading. One puzzle per block? would that
    work?
  - make sure things are properly asnychronous
- simple program: check if the answer is correct
  - could either double-check that all rules are followed, or
  - simply check against the answer in the database
- speed - is the copying back and forth really bad? Can it be async?
- output metrics to csv
- make command line args more usable (optparse or like)
  - verbosity flag? 
- clean up code by splitting into files

### Optional Extras
- larger puzzles going > single digits
- hard sudokus sometimes require guessing between two options for a cell - how to handle?
- generate a sudoku puzzle
  - started in module 8 work. Generates a grid of the appropriate size filled with
    ints in the right range, but it doesn't follow the rules of sudoku
- CI

[1] https://www.kaggle.com/bryanpark/sudoku/kernels
