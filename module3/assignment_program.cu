/**
 * Assignment 03 Program 
 * Sarah Helble
 * 9/17/17
 *
 * TODO: add description of what it does
 */

#include <stdio.h>
 
#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
 
char cpu_text[ARRAY_SIZE]; 
char cpu_key[ARRAY_SIZE];
 
__global__
 
void encrypt(char *text, char *text)
{
 		/* Calculate the current index */
 		const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
 
 		/* Create the cipherchar (addition of plaintext char and key char */
 		char cipherchar = text[thread_idx] + key[thread_idx];

 		/* Save back in text array */
 		text[thread_idx] = cipherchar; 
}
 
void main_sub()
{
 
 		/* Declare and allocate pointers for GPU based parameters */
 		char *gpu_text;
 		char *gpu_key;
 
 		cudaMalloc((void **)&gpu_text, ARRAY_SIZE_IN_BYTES);
 		cudaMalloc((void **)&gpu_key, ARRAY_SIZE_IN_BYTES);
 		
 		/* Copy the CPU memory to the GPU memory */
 		cudaMemcpy( cpu_text, gpu_text, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
 		cudaMemcpy( cpu_key, gpu_key, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
 
 		/* Designate the number of blocks and threads */
 		const unsigned int num_blocks = ARRAY_SIZE/16;
 		const unsigned int num_threads = ARRAY_SIZE/num_blocks;
 
 		/* Execute the encryption kernel */
 		encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key);
 }
 
 int main()
 {
 
 		main_sub();
 	
 		return EXIT_SUCCESS;
 }
 