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
#define NUM_ALPHA 127
 
unsigned int cpu_text[ARRAY_SIZE]; 
unsigned int cpu_key[ARRAY_SIZE];
unsigned int cpu_result[ARRAY_SIZE];
 
__global__ void encrypt(unsigned int *text, unsigned int *key, unsigned int *result)
{
 		/* Calculate the current index */
 		const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
 
 		/* Create the cipherchar (addition of plaintext char and key char */
 		result[idx] = (unsigned int) ( ( key[idx] + text[idx] ) % NUM_ALPHA );

 		/* TODO: Some of these values are unprintable. Make wrap more advanced */
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
 		unsigned int *gpu_text;
 		unsigned int *gpu_key;
 		unsigned int *gpu_result;
	
 		cudaMalloc((void **)&gpu_text, ARRAY_SIZE_IN_BYTES);
 		cudaMalloc((void **)&gpu_key, ARRAY_SIZE_IN_BYTES);
		cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
 		
 		/* Copy the CPU memory to the GPU memory */
 		cudaMemcpy( gpu_text, cpu_text, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
 		cudaMemcpy( gpu_key, cpu_key, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	
 		/* Designate the number of blocks and threads */
 		const unsigned int num_blocks = ARRAY_SIZE/16;
 		const unsigned int num_threads = ARRAY_SIZE/num_blocks;
 
 		/* Execute the encryption kernel */
 		encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);
 
 		/* Copy the GPU memory back to the CPU */
		cudaMemcpy( cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
 
 		/* Free the GPU memory */
 		cudaFree(gpu_text);
 		cudaFree(gpu_key);
		cudaFree(gpu_result);
 
 		/* Print the final result */
        printf("\nResults in ciphertext: \n");
        for(i = 0; i < ARRAY_SIZE; i++) 
        {
 			printf("%c ", (int)cpu_result[i]);	
 		}
 		printf("\n");                                     
 }
 
 int main()
 {
 	/* TODO: get input file, key file, array size and num blocks from command line */
 
        FILE *input_fp = fopen("input_text.txt", "r");
        FILE *key_fp = fopen("key.txt", "r");
        for(int i = 0; i < ARRAY_SIZE; i++) {
 		cpu_text[i] = fgetc(input_fp);
 		cpu_key[i] = fgetc(key_fp);  
	}
 
 	main_sub();
 	
 	return EXIT_SUCCESS;
 }
 
