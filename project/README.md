Class Project
Sarah Helble
1 Nov 2017

Sudoku Solver
-------------

Todo
----

### Required
- ingest sudoku puzzles from site
- solve a given sudoku puzzle
  - in the website I found [1], it's just a string of numbers, where 0 is empty cell
  - how to actually solve
    - could have one kernel solve by cell (check a cell for values it can hold. If only one, that's it)
    - and one kernel that solves by dimension (check a row for last values to fill, or values that can only fit in one place)
    - another kernel or host code to check if you're done
    - cycle through these until last returns success
- solve a lot of sudoku puzzles
  - how does this change the blocks/threading. One puzzle per block? would that work?
  - make things asnychronous
- simple program: check if the answer is correct
  - could either double-check that all rules are followed, or
  - simply check against the answer in the database
- timing data

### Optional
- larger puzzles going > single digits
- generate a sudoku puzzle
  - started in module 8 work. Generates a grid of the appropriate size filled with
    ints in the right range, but it doesn't follow the rules of sudoku
- CI

[1] https://www.kaggle.com/bryanpark/sudoku/kernels
