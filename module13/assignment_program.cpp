//
// Sarah Helble
// Assigment 13 Program
// 5 December 2017
//
// This program allows the user to enter a series of commands to be executed on
// an array of integers (initially filled with the value of their indices)
// The execution of the commands is alternated between two command queues, and
// events are used in order to guarantee that the commands are executed in the
// correct order, without one queue proceeding before the other is finished.
// Execute the program without any input to see usage.
//
// Pieces of boilerplate code adapted from:
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// and
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
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

enum command {ADD=1, SQUARE, TENFOLD, NEGATE, ADD_LEFT_PEER};

/**
 * Function to check and handle OpenCL errors
 * err is the returned status value to check
 * name is the name of the function, for printing to user
 */
inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Create an OpenCL context
 * Returns the created context. Exits program if failure
 */
cl_context CreateContext()
{
    cl_int errno;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // Select the first platform
    errno = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	checkErr(errno, "clGetPlatformIDs");
	if(numPlatforms <= 0) {
		checkErr(-1, "No Platforms Found");
	}

    // Attempt to create a GPU-based context. Use CPU if fails, but give warning
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errno);
    if (errno != CL_SUCCESS) {
        std::cout << "WARNING: Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errno);
        checkErr(errno, "clCreateContextFromType");
    }

    return context;
}

/**
 * Creates a command queue with the passed context.
 * Returns the command queue created, and sets device to the device used
 * Exits on failure
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

    // Choose the first available device
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &errno);
	checkErr(errno, "clCreateCommandQueue");

    *device = devices[0];
    delete [] devices;
    return queue;
}

/**
 * Create an OpenCL program from the passed kernel source file named in @filename,
 * using the passed context and device
 * Returns the program.
 * Exits on failure.
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

cl_kernel create_kernel_for_command(int command, cl_program program)
{
	cl_kernel kernel;

	switch(command) {
		case ADD:
			printf("Adding\n");
			kernel = clCreateKernel(program, "add", NULL);
			break;
		case SQUARE:
			printf("Squarin\n");
			kernel = clCreateKernel(program, "square", NULL);
			break;
		case TENFOLD:
			printf("Calculating x10\n");
			kernel = clCreateKernel(program, "tenfold", NULL);
			break;
		case NEGATE:
			printf("Negating\n");
			kernel = clCreateKernel(program, "negate", NULL);
			break;
		case ADD_LEFT_PEER:
			printf("Adding left peer\n");
			kernel = clCreateKernel(program, "add_left_peer", NULL);
			break;
		default:
			printf("Invalid command %d, ignoring\n", command);
			return NULL;
	}
	return kernel;
}

/**
 * Main program for driving the module 13 assignment Program
 */
int main(int argc, char** argv)
{
    cl_context context      = 0;
	cl_command_queue queue1 = 0;
	cl_command_queue queue2 = 0;
    cl_command_queue queue  = 0;
    cl_program program      = 0;
    cl_device_id device     = 0;
    cl_kernel kernel        = 0;
    cl_int errno;

	int *inputOutput;

    if (argc == 1) {
        std::cerr << "USAGE: " << argv[0] << " [execution_options]+ " << std::endl;
		std::cerr << "Execution options can be one or more of the following: " << std::endl;
		std::cerr << "\t1: add\n\t2: square\n\t3: tenfold (x10)" << std::endl;
		std::cerr << "\t4: negate\n\t5: add left peer" << std::endl;
		std::cerr << "\nFor example, '" << argv[0] << " 1 4 ' will result in the addition of each value to itself, followed by negation of the resultant value" << std::endl;

        return 1;
    }

    // Create an OpenCL context on first available platform
    context = CreateContext();

    // Create queues on the first device available on the created context
    queue1 = CreateCommandQueue(context, &device);
	queue2 = CreateCommandQueue(context, &device);

	inputOutput = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++) {
        inputOutput[i] = i;
    }

	// create a single buffer to cover all the input data
   cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
	   sizeof(int) * NUM_BUFFER_ELEMENTS, NULL, &errno);
   checkErr(errno, "clCreateBuffer");

    // Create OpenCL program
    program = CreateProgram(context, device, "assignment_kernels.cl");
	size_t globalWorkSize = NUM_BUFFER_ELEMENTS;

	cl_event copy_back_marker_event = NULL;
	cl_events command_events[argc-1] = NULL;

	for(int i = 1; i < argc; i++ ) {
		//divide onto separate queues for experiment
		queue = queue1;
		if(i % 2 == 0) queue = queue2;

		if(copy_back_marker_event != NULL) {
			clEnqueueWaitForEvents(queue, 1, &copy_back_marker_event);
		}

		// writing is waiting for event of other queue to complete
		errno = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0,
		  sizeof(int) * NUM_BUFFER_ELEMENTS, (void*)inputOutput, 0, NULL, NULL);

		// Create the kernel for the passed command and set kernel args
		int command = atoi(argv[i]);
		kernel = create_kernel_for_command(command, program);
		errno = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
		checkErr(errno, "clSetKernelArg()");

    	// Queue the kernel up for execution
		errno = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
			(const size_t*)&globalWorkSize, (const size_t*)NULL, 0, 0, &command_events[i-1]);
		clEnqueueBarrier(queue);

		// Read the output buffer back to the Host
		clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * NUM_BUFFER_ELEMENTS,
			(void*)inputOutput, 0, NULL, NULL);

		clEnqueueMarker(queue, &copy_back_marker_event);
	}

	clWaitForEvents(argc-1, command_events);
	cl_ulong start, end;
	double duration, duration_in_ms;

	for(int i = 0; i < argc - 1; i++) {
		errno = clGetEventProfilingInfo(command_events[i], CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
		checkErr(errno, "clGetEventProfilingInfo: start");

		errno = clGetEventProfilingInfo(command_events[i], CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
		checkErr(errno, "clGetEventProfilingInfo: end");

		duration = end - start;  // duration is in nanoseconds
		duration_in_ms = duration / 1000000;
		printf("command %d took %0.3fms\n", i+1, duration_in_ms);
	}

	for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++) {
		std::cout << " " << inputOutput[elems];
	}
	std::cout << std::endl;
	std::cout << "Program completed successfully" << std::endl;

    return 0;
}
