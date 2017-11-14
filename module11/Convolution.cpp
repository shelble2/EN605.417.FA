//
// Modified by Sarah Helble 10 Nov 2017 for Module 11 Assignment
// TODO:  Make sure filter is right, capture output
//

//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

#define MAX_INT 9

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_float inputSignal[inputSignalWidth][inputSignalHeight];

/*
 * Creating a function so that we don't need a hardcoded 49x49 array
 */
void fill_input_signal()
{
	for(int i = 0; i< inputSignalWidth; i++) {
		for(int j = 0; j < inputSignalHeight; j++) {
		  inputSignal[i][j] = (cl_float) (rand() % MAX_INT);
		}
	}
}


const unsigned int outputSignalWidth  = 7;
const unsigned int outputSignalHeight = 7;

cl_float outputSignal[outputSignalWidth][outputSignalHeight];


//Could probably be a kernel function instead
//Assumes square filter for math
cl_float *make_gradient_filter(unsigned int filterWidth, unsigned int filterHeight)
{
	cl_float *filter = (cl_float*) malloc(sizeof(cl_float)*filterWidth*filterHeight);

	cl_int half_width  = (cl_int) (filterWidth  / 2);
	cl_int half_height = (cl_int) (filterHeight / 2);
	cl_float increment = 100 / half_width-1;

	for(int i = 0; i < filterHeight; i++) {
		for(int j = 0; j < filterWidth; j++) {
			int hori_distance = abs(half_width-i);
			int vert_distance = abs(half_height-j);
			int greater = hori_distance;

			if(vert_distance > hori_distance)
				greater = vert_distance;

			cl_float gradient = (half_width-greater+1) * increment;
			filter[j+(i*filterWidth)] = gradient;
		}
	}
	return filter;
}

///
// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}



///
//	main_sub() for Convoloution example
//
int main_sub(unsigned int filterWidth, unsigned int filterHeight)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem filterBuffer;

	// Make host filter
	cl_float *filter = make_gradient_filter(filterWidth, filterHeight);

	fill_input_signal();
    // First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i],
            CL_DEVICE_TYPE_GPU,
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0)
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices,
				&deviceIDs[0],
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties,
		numDevices,
        deviceIDs,
		&contextCallback,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context,
		1,
		&src,
		&length,
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program,
			deviceIDs[0],
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
			buildLog,
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	filterBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * filterHeight * filterWidth,
		static_cast<void *>(filter),
		&errNum);
	checkErr(errNum, "clCreateBuffer(filter)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_float) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_float), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_float), &filterWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[1] = { outputSignalWidth * outputSignalHeight };
    const size_t localWorkSize[1]  = { 1 };

		cl_event event;

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
				globalWorkSize, localWorkSize, 0,	NULL, &event);
	checkErr(errNum, "clEnqueueNDRangeKernel");

	errNum = clEnqueueReadBuffer(queue, outputSignalBuffer, CL_TRUE, 0,
		sizeof(cl_float) * outputSignalHeight * outputSignalHeight,
		outputSignal, 0, NULL, NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

	errNum = clWaitForEvents(1, &event);
	checkErr(errNum, "clWaitForEvents");

	errNum = clFinish(queue);
	checkErr(errNum, "clFinish");

	cl_ulong start, end;
	double duration, duration_in_ms;

	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start),
		&start, NULL);
	checkErr(errNum, "clGetEventProfilingInfo: start");

	errNum = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end),
		&end, NULL);
	checkErr(errNum, "clGetEventProfilingInfo: end");

	duration = end - start;  // duration is in nanoseconds
	duration_in_ms = duration / 1000000;

	// Output the input buffer
	printf("Input Signal:\n");
	for (int y = 0; y < inputSignalHeight; y++) {
		for (int x = 0; x < inputSignalWidth; x++) {
			std::cout << inputSignal[x][y] << " ";
		}
		std::cout << std::endl;
	}

	// Output the filter
	printf("Filter:\n");
	for (int y = 0; y < filterHeight; y++) {
		for (int x = 0; x < filterWidth; x++) {
			std::cout << filter[x+(y*filterWidth)] << " ";
		}
		std::cout << std::endl;
	}

	// Output the result buffer
	printf("Output Signal:\n");
	for (int y = 0; y < outputSignalHeight; y++) {
		for (int x = 0; x < outputSignalWidth; x++) {
			std::cout << outputSignal[x][y] << " ";
		}
		std::cout << std::endl;
	}
	printf("The kernel executed in %0.3fms\n", duration_in_ms);

	std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}

//Test harness for module 11 assignment
int main(int argc, char** argv)
{
	printf("First run 7x7: \n");
	main_sub(7, 7);

	printf("Second run, 49x49: \n");
	main_sub(49, 49);

}
