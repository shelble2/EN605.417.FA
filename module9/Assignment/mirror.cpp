/**
 * Assignment 09 B
 * Sarah Helble
 * 30 Oct 2017
 *
 */

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <npp.h>
#include <Exceptions.h>

#include <ImagesNPP.h>
#include <ImagesCPU.h>
#include <ImageIO.h>

int main_sub(void)
{
  std::string image_fn = "Lena.pgm";
 
  // Construct the filename for the result
  std::string mirror_fn = image_fn;
  std::string::size_type dot = mirror_fn.rfind('.');
  if (dot != std::string::npos) {
    mirror_fn = mirror_fn.substr(0, dot);
  }

  mirror_fn += "_mirror.pgm";

  npp::ImageCPU_8u_C1 h_original;
 try { 
  npp::loadImage(image_fn, h_original);
 } catch (npp::Exception &rException) {
  std::cerr << "Error! Exception occurred: " << std::endl;
  std::cerr << rException << std::endl;
  exit(EXIT_FAILURE);
 }
  
  //Copy to device
  npp::ImageNPP_8u_C1 d_original(h_original);

  // create struct for ROI size
  NppiSize size_ROI = {(int)d_original.width() , (int)d_original.height() };

  //Declare a pointer for the result
  npp::ImageNPP_8u_C1 d_mirror(d_original.size());
 try { 
  NppStatus status = nppiMirror_8u_C1R(d_original.data(), 0, d_mirror.data(), 0, size_ROI, NPP_BOTH_AXIS);
 } catch (npp::Exception &rException) {
  std::cerr << "Error! Exception occurred: " << std::endl;
  std::cerr << rException << std::endl;
  exit(EXIT_FAILURE);
 }

  // Make host destination
  npp::ImageCPU_8u_C1 h_mirror(h_original.size());

  // Copy it back
  d_mirror.copyTo(h_mirror.data(), h_mirror.pitch());

  // Save it to file
  saveImage(mirror_fn, h_mirror);
  std::cout << "Saved image to file: " << mirror_fn << std::endl;

  nppiFree(d_original.data());
  nppiFree(d_mirror.data());
  nppiFree(h_original.data());
  nppiFree(h_mirror.data());

  exit(EXIT_SUCCESS);
}

int main(void) 
{
 main_sub();
}
