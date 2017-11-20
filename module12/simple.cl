//
// Modified by Sarah Helble 11.19.17 for Module 12 Assignment
//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This demonstrates taking an average of a subset of a buffer

__kernel void sub_average(__global int* const input,
						  __global int *const output,
						  const int sub_buf)
{
	int sum = 0;
	for(int i = 0; i < sub_buf; i++) {
		printf("\tinput[%d] is %d\n", i, input[i]);
		sum = sum + input[i];
	}
	output[0] = (int) (sum / sub_buf);
	printf("set output to %d\n", output[0]);
}
