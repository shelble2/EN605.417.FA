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
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void sub_average(__global * buffer)
{
	size_t id = get_global_id(0);
	//TODO: this is doing the average, but bad on the last and I don't think this
	// is how he wants it
	buffer[id] = (buffer[id] * buffer[id+1]) / 2;
}
