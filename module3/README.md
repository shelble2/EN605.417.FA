Helble Assignment 03
====================

ceasar_cipher.cu

./out <num_threads> <threads_per_block> <input_file> <key_file>

Performs a ceasar cipher for the first <num_threads> characters of <input_file>, using the first <num_threads> 
characters of <key_file> as the key.

Outputs the result to STDOUT

Size of the array (text string) and total number of threads used is specified by <num_threads>. 
Total number of threads per block is given by <threads_per_block>.
Thus, the total number of blocks will be <num_threads> / <threads_per_block>.
