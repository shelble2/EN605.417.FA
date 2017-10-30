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

int main(void)
{
  thrust::host_vector<int> H(NUM_ELEMENTS);
  thrust::device_vector<int> D(NUM_ELEMENTS);
  thrust::device_vector<int> M(NUM_ELEMENTS);
  thrust::device_vector<int> R(NUM_ELEMENTS);

  // Fill the device vector with all numbers between 0 and NUM_ELEMENTS
  thrust::sequence(D.begin(), D.end());

  // Fill the modulo vector with the target
  thrust::fill(M.begin(), M.end(), TARGET);

  // For each element in D, compute the modulus of i % TARGET (target stored in
  // M), and store result in R
  thrust::transform(D.begin(), D.end(), M.begin(), R.begin(), thrust::modulus<int>());

  // Copy the result back to the host
  H = R;

  // Print result
  for(int i = 0; i < H.size(); i++)
  {
    if(H[i] == 0) {
      std::cout << i << " is a multiple of " << TARGET << std::endl;
    }
  }
}
