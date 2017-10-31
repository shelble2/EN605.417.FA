/**
 * Assignment 09 B
 * Sarah Helble
 * 30 Oct 2017
 *
 */

#include <string.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <npp.h>

#include <Exceptions.h>
#include <ImagesNPP.h>
#include <ImagesCPU.h>
#include <ImageIO.h>

int mirror_sub(void)
{
  std::string image_fn = "Lena.pgm";

  // Construct the filename for the result
  std::string mirror_fn = image_fn;
  std::string::size_type dot = mirror_fn.rfind('.');
  if (dot != std::string::npos) {
    mirror_fn = mirror_fn.substr(0, dot);
  }
  mirror_fn += "_mirror.pgm";

  cudaEvent_t start, stop;
  float duration;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Recording from load to copy back 
  cudaEventRecord(start, 0);

  // Load the original image into host memory
  npp::ImageCPU_8u_C1 h_original;
  try {
    npp::loadImage(image_fn, h_original);
  } catch (npp::Exception &rException) {
    std::cerr << "Error! Exception occurred: " << std::endl;
    std::cerr << rException << std::endl;
    return EXIT_FAILURE;
 }

  // Copy to device
  npp::ImageNPP_8u_C1 d_original(h_original);

  NppiSize size_ROI = {(int)d_original.width() , (int)d_original.height() };

  // Declare a pointer for the result
  npp::ImageNPP_8u_C1 d_mirror(d_original.size());

  // Make a mirror image of the original
  try {
    NppStatus status = nppiMirror_8u_C1R(d_original.data(), d_original.pitch(), d_mirror.data(), d_mirror.pitch(), size_ROI, NPP_HORIZONTAL_AXIS);
    printf("Result of nppiMirror is %d\n", status);
  } catch (npp::Exception &rException) {
    std::cerr << "Error! Exception occurred: " << std::endl;
    std::cerr << rException << std::endl;
    return EXIT_FAILURE;
  }

  // Make host destination
  npp::ImageCPU_8u_C1 h_mirror(h_original.size());

  // Copy it back
  d_mirror.copyTo(h_mirror.data(), h_mirror.pitch());

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&duration, start, stop);
  printf("Elapsed Time: %f\n", duration);

  // Save it to file
  saveImage(mirror_fn, h_mirror);
  std::cout << "Saved image to file: " << mirror_fn << std::endl;

  // Free Everything
  nppiFree(d_original.data());
  nppiFree(d_mirror.data());
  nppiFree(h_original.data());
  nppiFree(h_mirror.data());

  return 0;
}

int main(void)
{
  printf("\nRun 1 of mirror program\n");
  mirror_sub();

  printf("\nRun 2 of mirror program\n");
  mirror_sub();

  printf("\n");
}
