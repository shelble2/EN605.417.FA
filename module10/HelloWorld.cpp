//
// Modified by Sarah Helble 4 Nov 2017 for Module 10 Assignment
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

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

enum Command{HELLO=0, ADD=1, SUBTRACT=2, MULTIPLY=3, DIVIDE=4, POWER=5};

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
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

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue, char *fn, char *kn,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    free(fn);
    free(kn);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

void choose_file_and_kernel(enum Command command, char **file_out, char **kernel_out)
{
  char *tmp_file = (char *)malloc(100*sizeof(char));
  char *tmp_kernel = (char *)malloc(100*sizeof(char));
  switch(command) {
    case HELLO:
      tmp_file = strdup("HelloWorld.cl");
      tmp_kernel = strdup("hello_kernel");
      break;
    case ADD:
      tmp_file = strdup("add.cl");
      tmp_kernel = strdup("add_kernel");
      break;
    case SUBTRACT:
      tmp_file = strdup("sub.cl");
      tmp_kernel = strdup("sub_kernel");
      break;
    case DIVIDE:
      tmp_file = strdup("div.cl");
      tmp_kernel = strdup("div_kernel");
      break;
    case MULTIPLY:
      tmp_file = strdup("mult.cl");
      tmp_kernel = strdup("mult_kernel");
      break;
    case POWER:
      tmp_file = strdup("pow.cl");
      tmp_kernel = strdup("pow_kernel");
      break;
    default:
      tmp_file = NULL;
      tmp_kernel = NULL;
  }

  *file_out = tmp_file;
  *kernel_out = tmp_kernel;
  return;
}

///
//	main_sub() for program
//
int main_sub(enum Command command)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;
    char *file_name = NULL;
    char *kernel_name = NULL;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }


    choose_file_and_kernel(command, &file_name, &kernel_name);
    if(!file_name || !kernel_name) {
      printf("Failed to find file name ane kernel name for command\n");
      Cleanup(context, commandQueue, file_name, kernel_name, program,
          kernel, memObjects);
      return -1;
    }

    program = CreateProgram(context, device, file_name);
    if (program == NULL) {
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, kernel_name, NULL);
    if (kernel == NULL) {
      std::cerr << "Failed to create kernel" << std::endl;
      Cleanup(context, commandQueue, file_name, kernel_name, program,
          kernel, memObjects);
      return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    cl_event event;
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &event);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, file_name, kernel_name, program,
            kernel, memObjects);
        return 1;
    }

    clWaitForEvents(1, &event);
    clFinish(commandQueue);

    cl_ulong start, end;
    double duration, duration_in_ms;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start),
      &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end),
      &end, NULL);

    duration = end - start;  // duration is in nanoseconds
    duration_in_ms = duration / 1000000;

    printf("The %s kernel executed in %0.3fms\n", kernel_name, duration_in_ms);

    // Output the result buffer
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, file_name, kernel_name, program,
        kernel, memObjects);

    return 0;
}

int main (int argc, char** argv)
{
  printf("\nArray size is %d\n", ARRAY_SIZE);

  /*  printf("\nFirst run; original HelloWorld\n");
  main_sub(HELLO);
  */
  printf("\nFirst run; ADD\n");
  main_sub(ADD);
  printf("\nSecond run; ADD\n");
  main_sub(ADD);

  printf("\nFirst run; SUBTRACT\n");
  main_sub(SUBTRACT);
  printf("\nSecond run; SUBTRACT\n");
  main_sub(SUBTRACT);

  printf("\nFirst run; MULTIPLY\n");
  main_sub(MULTIPLY);
  printf("\nSecond run; MULTIPLY\n");
  main_sub(MULTIPLY);

  printf("\nFirst run; DIVIDE\n");
  main_sub(DIVIDE);
  printf("\nSecond run; DIVIDE\n");
  main_sub(DIVIDE);

  printf("\nFirst run; POWER\n");
  main_sub(POWER);
  printf("\nSecond run; POWER\n");
  main_sub(POWER);
}
