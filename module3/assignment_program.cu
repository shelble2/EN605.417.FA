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
 
void encrypt(char *text, char *key)
{
 		/* Calculate the current index */
 		const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
 
 		/* Create the cipherchar (addition of plaintext char and key char */
 		char cipherchar = ( text[thread_idx] + key[thread_idx] );
 
		//TODO: need to wrap around here in order to have printable ciphertext. Can just use decimal value though
 	
 		/* Save back in text array */
 		text[thread_idx] = cipherchar; 
}
 
void main_sub()
{
 		int i = 0;
 
 		printf("Encrypting text: \n");
 		for(i = 0; i < ARRAY_SIZE; i++)
        {
        		printf("%c", cpu_text[i]);
        }
                                      
 		printf("\n With Key: \n");    
 		for(i = 0; i < ARRAY_SIZE; i++) 
 		{
 				printf("%c", cpu_key[i]);
        }
 
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
 
 		/* Copy the GPU memory back to the CPU 
 		cudaMemcpy( cpu_text, gpu_text, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
 		cudaMemcpy( cpu_key, gpu_key, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);*/

 		/* Free the GPU memory */
 		cudaFree(gpu_text);
 		cudaFree(gpu_key);
 
 		/* Print the final result */
        printf("\nResults in ciphertext: \n");
        for (i = 0; i < ARRAY_SIZE; i++) 
        {
 				printf("%d", cpu_text[i]);	
 		}
 		printf("\n");                                     
 }
 
 int main()
 {
 		/* TODO: get input file, key file, array size and num blocks from command line */
 
        FILE *input_fp = fopen("input_text.txt", "r");
        FILE *key_fp = fopen("input_text.txt", "r");
        for(int i = 0; i < ARRAY_SIZE; i++) {
 				cpu_text[i] = (char) fgetc(input_fp);
 				cpu_key[i] = (char) fgetc(key_fp);  
 		}
 
 		main_sub();
 	
 		return EXIT_SUCCESS;
 }
 