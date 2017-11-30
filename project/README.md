Class Project
Sarah Helble
14 Nov 2017

Sudoku Solver
-------------
Once finished, this program will be able to solve any sudoku puzzle read in from a file with efficiency.
Extras that I also want to include:
1. Check that an answer is correct
2. Ingest a bunch of sudoku puzzles from [1]
3. Generate sudoku puzzles of varying complexity
4. Possibly larger puzzles (double-digit sudokus?)

Current Status Summary
----------------------
Currently, this program solves a hardcoded sudoku puzzle (the first from [1])
and solves it by calling the solve_by_possibility kernel function until a host
function determines that the puzzle has been solved. It only uses the solve_by_possibility kernel. In my TODO, I identify other approaches that could
be alternated to achieve a faster solution.

It also needs to ingest the sudoku puzzle(s) from a file or [1] instead of using
the hardcoded one.

### TODO
- ingest sudoku puzzles from site
- solve a given sudoku puzzle
  - another kernel that solves by dimension (check a row for last values to fill)
  - switch to solving by constraint
  - hard sudokus sometimes require guessing between two options for a cell - how to handle?
- solve a lot of sudoku puzzles at once
  - how does this change the blocks/threading. One puzzle per block? would that work?
  - make things asnychronous
- simple program: check if the answer is correct
  - could either double-check that all rules are followed, or
  - simply check against the answer in the database
- speed - is the copying back and forth really bad? Can it be async?

### Optional Extras
- larger puzzles going > single digits
- generate a sudoku puzzle
  - started in module 8 work. Generates a grid of the appropriate size filled with
    ints in the right range, but it doesn't follow the rules of sudoku
- CI

[1] https://www.kaggle.com/bryanpark/sudoku/kernels
