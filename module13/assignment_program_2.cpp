//
// Sarah Helble
// Assigment 13 Program
// 5 December 2017
//
// Boilerplate code in part from:
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
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_BUFFER_ELEMENTS 16

/**
 * Function to check and handle OpenCL errors
 */
inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Create an OpenCL context on the first available platform using
 * either a GPU or CPU depending on what is available.
 */
cl_context CreateContext()
{
    cl_int errno;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
	// TODO: change this to be more like simple_async.cpp
    errno = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	checkErr(errno, "clGetPlatformIDs");
	if(numPlatforms <= 0) {
		checkErr(-1, "No Platforms Found");
	}

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errno);
    if (errno != CL_SUCCESS) {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errno);
        checkErr(errno, "clCreateContextFromType");
    }

    return context;
}

/**
 *  Create a command queue on the first device available on the
 *  context
 */
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errno;
    cl_device_id *devices;
    cl_command_queue queue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errno = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	checkErr(errno, "clGetContextInfo");
    if (deviceBufferSize <= 0) {
		checkErr(-1, "No devices found");
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errno = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    checkErr(errno, "clGetContextInfo");

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    queue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (queue == NULL) {
		checkErr(-1, "clCreateCommandQueue");
    }

    *device = devices[0];
    delete [] devices;
    return queue;
}

/**
 *  Create an OpenCL program from the kernel source file
 */
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errno;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
		checkErr(-1, "kernelFile.is_open()");
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
    if (program == NULL) {
		checkErr(-1, "clCreateProgramWithSource");
    }

    errno = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errno != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

int main(int argc, char** argv)
{
    cl_context context     = 0;
    cl_command_queue queue = 0;
    cl_program program     = 0;
    cl_device_id device    = 0;
    cl_kernel kernel       = 0;
    cl_int errno;

	int *inputOutput1;

    if (argc != 1) {
        std::cerr << "USAGE: " << argv[0] << std::endl;
        return 1;
    }

    // Create an OpenCL context on first available platform
    context = CreateContext();

    // Create a command-queue on the first device available
    // on the created context
    queue = CreateCommandQueue(context, &device);

	inputOutput = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++) {
        inputOutput[i] = i;
    }

	// create a single buffer to cover all the input data
   cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
	   sizeof(int) * NUM_BUFFER_ELEMENTS, NULL, &errno);
   checkErr(errno, "clCreateBuffer");

    // Create OpenCL program
    program = CreateProgram(context, device, "simple.cl");

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "cube", NULL);

	errno = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
    checkErr(errno, "clSetKernelArg(cube)");

	errno = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
      sizeof(int) * NUM_BUFFER_ELEMENTS, (void*)inputOutput, 0, NULL, NULL);

	cl_event event;
	size_t globalWorkSize = NUM_BUFFER_ELEMENTS;

    // Queue the kernel up for execution
	errno = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
		(const size_t*)&globalWorkSize, (const size_t*)NULL, 0, 0, &event);

    // Read the output buffer back to the Host
	clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS,
            (void*)inputOutput, 0, NULL, NULL);

	for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++) {
		std::cout << " " << inputOutput[elems];
	}
	std::cout << std::endl;
	std::cout << "Program completed successfully" << std::endl;

    return 0;
}
