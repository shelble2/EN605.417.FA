//
// Modified by Sarah Helble for Module 12 Assignment 11.19.2017
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

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16
#define SUB_BUF 4
#define NUM_SUB_BUF NUM_BUFFER_ELEMENTS / SUB_BUF

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

// Display ouyput in rows
void display_output(int *inputOutput)
{
	for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++) {
		std::cout << " " << inputOutput[elems];
	}
	std::cout << std::endl;
}

//
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
	cl_int errNum;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
	cl_context context;
	cl_program program;
	std::vector<cl_mem> buffers;
	std::vector<cl_mem> output_buffers;
	int * inputOutput;

	int platform = DEFAULT_PLATFORM;

	std::cout << "Buffer and sub-buffer Example for averaging" << std::endl;

	for (int i = 1; i < argc; i++) {
		std::string input(argv[i]);

		if (!input.compare("--platform")) {
			input = std::string(argv[++i]);
			std::istringstream buffer(input);
			buffer >> platform;
		} else {
			std::cout << "usage: --platform n " << std::endl;
			return 0;
		}
	}

	// First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

	std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr((errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

	// Load the kernel file to source string
	std::ifstream srcFile("simple.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

	std::string srcProg(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Display info of the platform we're using
	DisplayPlatformInfo( platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");

	// Get device information
	errNum = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND) {
		checkErr(errNum, "clGetDeviceIDs");
	}
	std::cout << "Number of devices: \t" << numDevices << std::endl;

	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
	errNum = clGetDeviceIDs( platformIDs[platform], CL_DEVICE_TYPE_ALL, numDevices, &deviceIDs[0], NULL);
	checkErr(errNum, "clGetDeviceIDs");

	// make the context
	cl_context_properties contextProperties[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[platform],
		0 };

	context = clCreateContext( contextProperties, numDevices, deviceIDs, NULL, NULL, &errNum);
	checkErr(errNum, "clCreateContext");

    // Create program from source
	program = clCreateProgramWithSource(context, 1, &src, &length, &errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram( program, numDevices, deviceIDs, "-I.", NULL,  NULL);
	if (errNum != CL_SUCCESS) {
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, deviceIDs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in OpenCL C source: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}

	cl_kernel kernel = clCreateKernel( program, "sub_average", &errNum);
	checkErr(errNum, "clCreateKernel(sub_average)");

	// Use the first device
	cl_command_queue queue = clCreateCommandQueue(context, deviceIDs[0], CL_QUEUE_PROFILING_ENABLE, &errNum);
	checkErr(errNum, "clCreateCommandQueue");

	// create host buffer
	inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
	for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++) {
		inputOutput[i] = i;
	}

	//TODO: timing, and should really be 2x2, not 4x1


	// create a single device buffer to cover all the input data
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(int) * NUM_BUFFER_ELEMENTS,
		static_cast<void *>(inputOutput), &errNum);
	checkErr(errNum, "clCreateBuffer");

	cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(float) * SUB_BUF, NULL, &errNum);
	checkErr(errNum, "clCreateBuffer");

	// create a sub buffer and output sub buffer for each region of data
	for (unsigned int i = 0; i < NUM_SUB_BUF; i++) {
		cl_buffer_region region = {
			i * SUB_BUF * sizeof(int),
			SUB_BUF * sizeof(int) };

		cl_mem sub_buffer = clCreateSubBuffer(buffer, CL_MEM_READ_ONLY,
			CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "clCreateSubBuffer");
		buffers.push_back(sub_buffer);

		cl_buffer_region output_region = {
			i * sizeof(float),
			sizeof(float),
		};

		cl_mem output_sub_buffer = clCreateSubBuffer(output_buffer, CL_MEM_WRITE_ONLY,
			CL_BUFFER_CREATE_TYPE_REGION, &output_region, &errNum);
		checkErr(errNum, "clCreateSubBuffer");
		output_buffers.push_back(output_sub_buffer);
	}

	cl_int sub_buf_sz = SUB_BUF;
	cl_event *events[SUB_BUF];

	//Now enqueue all the kernels
	for(unsigned int i = 0; i < NUM_SUB_BUF; i++) {
		errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
		checkErr(errNum, "clSetKernelArg(sub_average)");
		errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_buffers[i]);
		checkErr(errNum, "clSetKernelArg(sub_average)");
		errNum |= clSetKernelArg(kernel, 2, sizeof(cl_int), &sub_buf_sz);

		const size_t globalWorkSize[1] = { NUM_BUFFER_ELEMENTS };
	    const size_t localWorkSize[1]  = { SUB_BUF };

		errNum = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
			(const size_t*)NUM_BUFFER_ELEMENTS, (const size_t*)NULL, 0, 0, &events[i]);
		checkErr(errNum, "clEnqueueNDRangeKernel");
	}

	clWaitForEvents(events.size(), &events[0]);

	// Read back computed data
	clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
		sizeof(float) * SUB_BUF, (void*)inputOutput,
		0, NULL, NULL);

	display_output(inputOutput);

	std::cout << "Program completed successfully" << std::endl;

	return 0;
}
