//
// Assignment 13
// 5 December 2017
// Sarah Helble
// Boilerplate code partially comes from:
//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Creates the context for this program
 */
 cl_context create_context()
 {
	 cl_int errno;
	 cl_uint num_platforms;
	 cl_uint num_devices;
	 cl_platform_id *platform_ids = NULL;
	 cl_device_id *device_ids = NULL;

	 int platform = DEFAULT_PLATFORM;

	 // Get the number of platforms for allocation
     errno = clGetPlatformIDs(0, NULL, &num_platforms);
     checkErr(errno, "clGetPlatformIDs");
	 if(num_platforms <= 0) {
		 checkErr(-1, "Invalid Number of Platforms");
	 }

     platform_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * num_platforms);

     std::cout << "Found "<< num_platforms << " platforms" << std::endl;

	 // Get all of the platform IDs
     errno = clGetPlatformIDs(num_platforms, platform_ids, NULL);
     checkErr(errno, "clGetPlatformIDs");

	 cl_context_properties contextProperties[] =
	 {
	 	CL_CONTEXT_PLATFORM,
	 	(cl_context_properties)platform_ids[platform],
	 	0
	 };

	DisplayPlatformInfo(platform_ids[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");

	// Get the number of devices for allocation
	errno = clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	checkErr(errno, "clGetDeviceIDs");

	device_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * num_devices);
	//Get all of the devices
	errno = clGetDeviceIDs(platform_ids[platform], CL_DEVICE_TYPE_ALL, num_devices, &device_ids[0], NULL);
	checkErr(errno, "clGetDeviceIDs");

	cl_context context = clCreateContext(contextProperties, num_devices, device_ids, NULL, NULL, &errno);
	checkErr(errno, "clCreateContext");

	return context;
 }

int main(int argc, char** argv)
{
    cl_int errno;
    cl_context context;
    cl_program program0;
    cl_program program1;
	int * inputOutput0;
    int * inputOutput1;

	context = create_context();

	std::ifstream srcFile("simple.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

    // Create program from source
    program0 = clCreateProgramWithSource(context, 1, &src, &length, &errno);
    checkErr(errno, "clCreateProgramWithSource");

     // Create program from source
    program1 = clCreateProgramWithSource(context, 1, &src, &length, &errno);
    checkErr(errno, "clCreateProgramWithSource");

    // Build program
    errno = clBuildProgram(program0, 0, NULL, NULL, NULL, NULL);

    // Build program
    errno |= clBuildProgram(program1, 0, NULL, NULL, NULL, NULL);
	checkErr(errno, "clBuildProgram");

    // create buffers
    inputOutput0 = new int[NUM_BUFFER_ELEMENTS];
	inputOutput1 = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++) {
        inputOutput0[i] = i;
		inputOutput1[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem buffer0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS, NULL, &errno);
    checkErr(errno, "clCreateBuffer");

     // create a single buffer to cover all the input data
    cl_mem buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS, NULL, &errno);
    checkErr(errno, "clCreateBuffer");

    // Create command queues
    InfoDevice<cl_device_type>::display(device_ids[0], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");

    cl_command_queue queue0 = clCreateCommandQueue(context, device_ids[0], 0, &errno);
    checkErr(errno, "clCreateCommandQueue");

    cl_command_queue queue1 = clCreateCommandQueue(context, device_ids[0], 0, &errno);
    checkErr(errno, "clCreateCommandQueue");

    cl_kernel kernel0 = clCreateKernel(program0, "square", &errno);
    checkErr(errno, "clCreateKernel(square)");

    cl_kernel kernel1 = clCreateKernel(program1, "cube", &errno);
    checkErr(errno, "clCreateKernel(cube)");

    errno = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&buffer0);
    checkErr(errno, "clSetKernelArg(square)");

    errno = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&buffer1);
    checkErr(errno, "clSetKernelArg(cube)");

    // Write input data
    errno = clEnqueueWriteBuffer(queue0, buffer0, CL_TRUE, 0,
      sizeof(int) * NUM_BUFFER_ELEMENTS,
      (void*)inputOutput0, 0, NULL, NULL);

    errno = clEnqueueWriteBuffer(queue1, buffer1, CL_TRUE, 0,
      sizeof(int) * NUM_BUFFER_ELEMENTS,
      (void*)inputOutput1, 0, NULL, NULL);

    std::vector<cl_event> events;
    // call kernel for each device
    cl_event event0;
	cl_event event1;

    size_t gWI = NUM_BUFFER_ELEMENTS;

    errno = clEnqueueNDRangeKernel(queue0, kernel0, 1, NULL, (const size_t*)&gWI,
      (const size_t*)NULL, 0, 0, &event0);

 	errno = clEnqueueMarker(queue1, &event1);

    errno = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, (const size_t*)&gWI,
      (const size_t*)NULL, 0, 0, &event0);

 	//Wait for queue 1 to complete before continuing on queue 0
 	errno = clEnqueueBarrier(queue0);
 	errno = clEnqueueWaitForEvents(queue0, 1, &event1);

 	// Read back computed data
   	clEnqueueReadBuffer(queue0, buffer0, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS,
            (void*)inputOutput0, 0, NULL, NULL);

   	clEnqueueReadBuffer(queue1, buffer1, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS,
            (void*)inputOutput1, 0, NULL, NULL);

    // Display output in rows
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++) {
     std::cout << " " << inputOutput0[elems];
    }
    std::cout << std::endl;

    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++) {
     std::cout << " " << inputOutput1[elems];
    }
    std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
