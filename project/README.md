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

Current Status
--------------
Currently, this program solves a hardcoded sudoku puzzle (the first from [1]) and solves it by calling the solve_by_possibility 
kernel function 4x. It doesn't stop because it knows it's done, it stops because I hardcoded it to 4 knowing that that's how
many it takes. It only uses the solve_by_possibility kernel. In my TODO, I identify other approaches that could be alternated to
achieve a faster solution.

It also needs to ingest the sudoku puzzle(s) from a file or [1] instead of using the hardcoded one, and needs to be able to 
identify when its finished.

BUT major success that its actually solving the puzzle! All should go smoothly from here. 

Todo
----

### Required
- ingest sudoku puzzles from site
- solve a given sudoku puzzle
  - in the website I found [1], it's just a string of numbers, where 0 is empty cell
  - how to actually solve
    /- could have one kernel solve by cell (check a cell for values it can hold. If only one, that's it)
    - and one kernel that solves by dimension (check a row for last values to fill)
    - another kernel or host code to check if you're done
    - cycle through these until last returns success
- solve a lot of sudoku puzzles
  - how does this change the blocks/threading. One puzzle per block? would that work?
  - make things asnychronous
- simple program: check if the answer is correct
  - could either double-check that all rules are followed, or
  - simply check against the answer in the database
/- timing data

### Optional
- larger puzzles going > single digits
- generate a sudoku puzzle
  - started in module 8 work. Generates a grid of the appropriate size filled with
    ints in the right range, but it doesn't follow the rules of sudoku
- CI

[1] https://www.kaggle.com/bryanpark/sudoku/kernels
