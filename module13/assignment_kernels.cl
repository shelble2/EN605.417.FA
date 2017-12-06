//
// Sarah Helble
// Assignment 13
// 5 December 2017
// The square kernel is taken from simple.cl. 

// assignment_kernels.cl
// This file houses mutiple simple kernels to be used in the
// assignment_program.cpp

__kernel void square(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}

__kernel void tenfold(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * 10;
}

__kernel void add(__global *buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] + buffer[id]
}

__kernel void negative(__global *buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] - (buffer[id] * 2);
}
