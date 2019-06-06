/*
 * backward.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
static const bool tbl[6][16] = {
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X
bool CNN::Backward_output(int index)
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_OUT], 0, sizeof(cl_mem), &Forward_out_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_OUT], 1, sizeof(cl_mem), &cl_label_input_train) ||
		clSetKernelArg(Backward_kernel[BACKWARD_OUT], 2, sizeof(cl_mem), &Backward_out_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_OUT], 3, sizeof(cl_int), &index) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_output arguments.\n");
		return false;
	}
	// size_t local[3];
	size_t global[1] = {num_neuron_output_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_OUT], 1, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_output. Error Code=%d\n", err); 
		return false;
	}
	clFinish(command_queue);

	return true;
}


bool CNN::Backward_C5()
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_C5], 0, sizeof(cl_mem), &Backward_out_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C5], 1, sizeof(cl_mem), &Forward_C5_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C5], 2, sizeof(cl_mem), &Forward_weight[FORWARD_OUT]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C5], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_OUT]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C5], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_OUT]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C5], 5, sizeof(cl_mem), &Backward_C5_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_C5 arguments.\n");
		return false;
	}
	// size_t local[3];
	size_t global[1] = {num_map_C5_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_C5], 1, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_C5. Error Code=%d\n", err); 
		return false;
	}
	clFinish(command_queue);

	return true;
}

bool CNN::Backward_S4()
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_S4], 0, sizeof(cl_mem), &Backward_C5_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S4], 1, sizeof(cl_mem), &Forward_S4_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S4], 2, sizeof(cl_mem), &Forward_weight[FORWARD_C5]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S4], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_C5]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S4], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_C5]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S4], 5, sizeof(cl_mem), &Backward_S4_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_S4 arguments.\n");
		return false;
	}
	size_t local[3] = {1,5,5} ;
	size_t global[3] = {num_map_S4_CNN,5,5};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_S4], 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_S4. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Backward_C3()
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_C3], 0, sizeof(cl_mem), &Backward_S4_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C3], 1, sizeof(cl_mem), &Forward_C3_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C3], 2, sizeof(cl_mem), &Forward_weight[FORWARD_S4]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C3], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_S4]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C3], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_S4]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C3], 5, sizeof(cl_mem), &Backward_C3_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_C3 arguments.\n");
		return false;
	}
	size_t local[3]={1,5,5};
	size_t global[3] = {num_map_C3_CNN,5,5};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_C3], 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_C3. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Backward_S2()
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_S2], 0, sizeof(cl_mem), &Backward_C3_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S2], 1, sizeof(cl_mem), &Forward_S2_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S2], 2, sizeof(cl_mem), &Forward_weight[FORWARD_C3]) ||
		// clSetKernelArg(Backward_kernel[BACKWARD_S2], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_C3]) ||
		// clSetKernelArg(Backward_kernel[BACKWARD_S2], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_C3]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_S2], 3, sizeof(cl_mem), &Backward_S2_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_S2 arguments.\n");
		return false;
	}
	// size_t local[1]= {1};
	size_t global[3] = {14,14,6};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_S2], 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_S2. Error Code=%d\n", err); 
		return false;
	}

	if (clSetKernelArg(Backward_kernel_s2_weight, 0, sizeof(cl_mem), &Backward_C3_mem) ||
		clSetKernelArg(Backward_kernel_s2_weight, 1, sizeof(cl_mem), &Forward_S2_mem) ||
		// clSetKernelArg(Backward_kernel_s2_weight, 2, sizeof(cl_mem), &Forward_weight[FORWARD_C3]) ||
		clSetKernelArg(Backward_kernel_s2_weight, 2, sizeof(cl_mem), &Backward_weight[BACKWARD_C3]) != CL_SUCCESS)
		// clSetKernelArg(Backward_kernel_s2_weight, 4, sizeof(cl_mem), &Backward_bias[BACKWARD_C3]) ||
		// clSetKernelArg(Backward_kernel_s2_weight, 3, sizeof(cl_mem), &Backward_S2_mem) 
	{
		printf("Unable to set kernel Backward_S2 arguments.\n");
		return false;
	}
	// size_t local[1]= {1};
	size_t global1[3] = {25,16,6};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel_s2_weight, 3, NULL, global1, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_S2. Error Code=%d\n", err); 
		return false;
	}

	if (clSetKernelArg(Backward_kernel_s2_bias, 0, sizeof(cl_mem), &Backward_C3_mem) ||
		// clSetKernelArg(Backward_kernel_s2_bias, 1, sizeof(cl_mem), &Forward_S2_mem) ||
		// clSetKernelArg(Backward_kernel_s2_bias, 2, sizeof(cl_mem), &Forward_weight[FORWARD_C3]) ||
		// clSetKernelArg(Backward_kernel_s2_bias, 2, sizeof(cl_mem), &Backward_weight[BACKWARD_C3])
		clSetKernelArg(Backward_kernel_s2_bias, 1, sizeof(cl_mem), &Backward_bias[BACKWARD_C3])  != CL_SUCCESS)
		// clSetKernelArg(Backward_kernel_s2_bias, 3, sizeof(cl_mem), &Backward_S2_mem) 
	{
		printf("Unable to set kernel Backward_S2 arguments.\n");
		return false;
	}
	// size_t local[1]= {1};
	size_t global2[1] = {16};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel_s2_bias, 1, NULL, global2, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_S2. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Backward_C1()
{
	if (clSetKernelArg(Backward_kernel[BACKWARD_C1], 0, sizeof(cl_mem), &Backward_S2_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C1], 1, sizeof(cl_mem), &Forward_C1_mem) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C1], 2, sizeof(cl_mem), &Forward_weight[FORWARD_S2]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C1], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_S2]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C1], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_S2]) ||
		clSetKernelArg(Backward_kernel[BACKWARD_C1], 5, sizeof(cl_mem), &Backward_C1_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_C1 arguments.\n");
		return false;
	}
	//[6,14,14]
	//[1,14,14]
	size_t global[3] = {6,14,14};
	size_t local[3] ={1,14,14};	

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_C1], 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_C1. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Backward_input(int index)
{
	// if (clSetKernelArg(Backward_kernel[BACKWARD_IN], 0, sizeof(cl_mem), &Backward_C1_mem) ||
	// 	clSetKernelArg(Backward_kernel[BACKWARD_IN], 1, sizeof(cl_mem), &cl_data_input_train) ||
	// 	clSetKernelArg(Backward_kernel[BACKWARD_IN], 2, sizeof(cl_mem), &Forward_weight[FORWARD_C1]) ||
	// 	// clSetKernelArg(Backward_kernel[BACKWARD_IN], 3, sizeof(cl_mem), &Backward_weight[BACKWARD_C1]) ||
	// 	// clSetKernelArg(Backward_kernel[BACKWARD_IN], 4, sizeof(cl_mem), &Backward_bias[BACKWARD_C1]) ||
	// 	clSetKernelArg(Backward_kernel[BACKWARD_IN], 3, sizeof(cl_mem), &Backward_in_mem) || 
	// 	clSetKernelArg(Backward_kernel[BACKWARD_IN], 4, sizeof(cl_int), &index) != CL_SUCCESS)
	// {
	// 	printf("Unable to set kernel Backward_input arguments.\n");
	// 	return false;
	// }
	// // size_t local[3];
	// size_t global[2] = {32,32};

	// err = clEnqueueNDRangeKernel(command_queue, Backward_kernel[BACKWARD_IN], 2, NULL, global, NULL /*local*/, 0, NULL, NULL);
	// if (err != CL_SUCCESS)
	// {
	// 	printf("Unable to enqueue kernel command Backward_input. Error Code=%d\n", err); 
	// 	return false;
	// }

	if (clSetKernelArg(Backward_kernel_input_weight, 0, sizeof(cl_mem), &Backward_C1_mem) ||
		clSetKernelArg(Backward_kernel_input_weight, 1, sizeof(cl_mem), &cl_data_input_train) ||
		// clSetKernelArg(Backward_kernel_input_weight, 2, sizeof(cl_mem), &Forward_weight[FORWARD_C1]) ||
		clSetKernelArg(Backward_kernel_input_weight, 2, sizeof(cl_mem), &Backward_weight[BACKWARD_C1]) ||
		// clSetKernelArg(Backward_kernel_input_weight, 4, sizeof(cl_mem), &Backward_bias[BACKWARD_C1]) ||
		// clSetKernelArg(Backward_kernel_input_weight, 3, sizeof(cl_mem), &Backward_in_mem) || 
		clSetKernelArg(Backward_kernel_input_weight, 3, sizeof(cl_int), &index) != CL_SUCCESS)
	{
		printf("Unable to set kernel Backward_input arguments.\n");
		return false;
	}
	size_t global[3] = {6,28,28};
	// size_t global[3] = {6,5,5};
	size_t local[3] = {1,28,28};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel_input_weight, 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_input. Error Code=%d\n", err); 
		return false;
	}

	if (clSetKernelArg(Backward_kernel_input_bias, 0, sizeof(cl_mem), &Backward_C1_mem) ||
		// clSetKernelArg(Backward_kernel_input_bias, 1, sizeof(cl_mem), &cl_data_input_train) ||
		// clSetKernelArg(Backward_kernel_input_bias, 2, sizeof(cl_mem), &Forward_weight[FORWARD_C1]) ||
		// clSetKernelArg(Backward_kernel_input_bias, 3, sizeof(cl_mem), &Backward_weight[BACKWARD_C1]) ||
		clSetKernelArg(Backward_kernel_input_bias, 1, sizeof(cl_mem), &Backward_bias[BACKWARD_C1])!= CL_SUCCESS)
		// clSetKernelArg(Backward_kernel_input_bias, 3, sizeof(cl_mem), &Backward_in_mem) || 
		// clSetKernelArg(Backward_kernel_input_bias, 4, sizeof(cl_int), &index) 
	{
		printf("Unable to set kernel Backward_input arguments.\n");
		return false;
	}
	// size_t local[3];
	size_t global2[1] = {6};

	err = clEnqueueNDRangeKernel(command_queue, Backward_kernel_input_bias, 1, NULL, global2, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Backward_input. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

