/**
 * Module 9 Thrust Assignment
 * Sarah Helble
 * 30 Oct 2017
 *
 * This program take a list of integers as input and uses the Thrust library
 * to determine whether each is a multiple of the passed integer.
 *
 * Usage: ./a.out
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <iostream>

#define NUM_ELEMENTS 1048
#define TARGET 7

void main_sub()
{
  thrust::host_vector<int> H(NUM_ELEMENTS);
  thrust::device_vector<int> D(NUM_ELEMENTS);
  thrust::device_vector<int> M(NUM_ELEMENTS);
  thrust::device_vector<int> R(NUM_ELEMENTS);

  cudaEvent_t start, stop;
	float duration;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  /* Recording from fill to copy back */
	cudaEventRecord(start, 0);

  // Fill the device vector with all numbers between 0 and NUM_ELEMENTS
  thrust::sequence(D.begin(), D.end());

  // Fill the modulo vector with the target
  thrust::fill(M.begin(), M.end(), TARGET);

  // For each element in D, compute the modulus of i % TARGET (target stored in
  // M), and store result in R
  thrust::transform(D.begin(), D.end(), M.begin(), R.begin(), thrust::modulus<int>());

  // Copy the result back to the host
  H = R;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);

  // Print result
  printf("%d elements\n", NUM_ELEMENTS);
  printf("Multiples of %d:\n", TARGET);
  for(int i = 0; i < H.size(); i++)
  {
    if(H[i] == 0) {
      std::cout << " " << i << " " << std::endl;
    }
  }
  printf("Elapsed Time: %f", duration);

}

int main(void)
{

  printf("Run 1 of Thrust Program\n");
  main_sub();

  printf("Run 2 of Thrust Program\n");
  main_sub();
}
