/*
 * forward.cpp
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


bool CNN::Forward_C1(int index, cl_mem & Forward_in_mem0)
{
	// TODO： 添加参数指示输入数据的地址
	if (clSetKernelArg(Forward_kernel[FORWARD_C1], 0, sizeof(cl_mem), &Forward_in_mem0) ||
		clSetKernelArg(Forward_kernel[FORWARD_C1], 1, sizeof(cl_mem), &Forward_weight[FORWARD_C1]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C1], 2, sizeof(cl_mem), &Forward_bias[FORWARD_C1]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C1], 3, sizeof(cl_mem), &Forward_C1_mem) ||
		clSetKernelArg(Forward_kernel[FORWARD_C1], 4, sizeof(cl_int), &index) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_C1 arguments.\n");
		return false;
	}
	size_t local[3] = {1, 7,7};
	size_t global[3] = {num_map_C1_CNN, height_image_C1_CNN,width_image_C1_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_C1], 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_C1. Error Code=%d\n", err); 
		return false;
	}

	// errs[1] = clEnqueueReadBuffer(command_queue, Forward_C1_out, CL_TRUE,
	// 						 0, num_neuron_C1_CNN*sizeof(cl_float), neuron_C1, 0, NULL, NULL);
	// if (errs[1] != CL_SUCCESS)
	// {
	// 	printf("Error enqueuing read buffer command Forward_C1. Error Code=%d\n", errs[1]);
	// 	return false;
	// }
	clFinish(command_queue);
	return true;
}


bool CNN::Forward_S2()
{
	if (clSetKernelArg(Forward_kernel[FORWARD_S2], 0, sizeof(cl_mem), &Forward_C1_mem) ||
		clSetKernelArg(Forward_kernel[FORWARD_S2], 1, sizeof(cl_mem), &Forward_weight[FORWARD_S2]) ||
		clSetKernelArg(Forward_kernel[FORWARD_S2], 2, sizeof(cl_mem), &Forward_bias[FORWARD_S2]) ||
		clSetKernelArg(Forward_kernel[FORWARD_S2], 3, sizeof(cl_mem), &Forward_S2_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_S2 arguments.\n");
		return false;
	}
	// size_t local[3];
	size_t global[3] = {num_map_S2_CNN, height_image_S2_CNN,width_image_S2_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_S2], 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_S2. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Forward_C3()
{
	if (clSetKernelArg(Forward_kernel[FORWARD_C3], 0, sizeof(cl_mem), &Forward_S2_mem) ||
		clSetKernelArg(Forward_kernel[FORWARD_C3], 1, sizeof(cl_mem), &Forward_weight[FORWARD_C3]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C3], 2, sizeof(cl_mem), &Forward_bias[FORWARD_C3]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C3], 3, sizeof(cl_mem), &Forward_C3_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_C3 arguments.\n");
		return false;
	}

	size_t local[3]={1,10,10};
	size_t global[3] = {num_map_C3_CNN, height_image_C3_CNN,width_image_C3_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_C3], 3, NULL, global, local /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_C3. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Forward_S4()
{
	if (clSetKernelArg(Forward_kernel[FORWARD_S4], 0, sizeof(cl_mem), &Forward_C3_mem) ||
		clSetKernelArg(Forward_kernel[FORWARD_S4], 1, sizeof(cl_mem), &Forward_weight[FORWARD_S4]) ||
		clSetKernelArg(Forward_kernel[FORWARD_S4], 2, sizeof(cl_mem), &Forward_bias[FORWARD_S4]) ||
		clSetKernelArg(Forward_kernel[FORWARD_S4], 3, sizeof(cl_mem), &Forward_S4_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_S4 arguments.\n");
		return false;
	}
	// size_t local[3];
	size_t global[3] = {num_map_S4_CNN, height_image_S4_CNN,width_image_S4_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_S4], 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_S4. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Forward_C5()
{
	if (clSetKernelArg(Forward_kernel[FORWARD_C5], 0, sizeof(cl_mem), &Forward_S4_mem) ||
		clSetKernelArg(Forward_kernel[FORWARD_C5], 1, sizeof(cl_mem), &Forward_weight[FORWARD_C5]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C5], 2, sizeof(cl_mem), &Forward_bias[FORWARD_C5]) ||
		clSetKernelArg(Forward_kernel[FORWARD_C5], 3, sizeof(cl_mem), &Forward_C5_mem) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_C5 arguments.\n");
		return false;
	}

	// size_t local[3];
	size_t global[3] = {num_map_C5_CNN, height_image_C5_CNN,width_image_C5_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_C5], 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_C5. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	return true;
}

bool CNN::Forward_output()
{
	if ((errs[0] = clSetKernelArg(Forward_kernel[FORWARD_OUT], 0, sizeof(cl_mem), &Forward_C5_mem)) ||
		(errs[1] = clSetKernelArg(Forward_kernel[FORWARD_OUT], 1, sizeof(cl_mem), &Forward_weight[FORWARD_OUT])) ||
		(errs[2] = clSetKernelArg(Forward_kernel[FORWARD_OUT], 2, sizeof(cl_mem), &Forward_bias[FORWARD_OUT])) ||
		(errs[3] = clSetKernelArg(Forward_kernel[FORWARD_OUT], 3, sizeof(cl_mem), &Forward_out_mem)) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_OUT arguments, err code: %d %d %d %d \n",errs[0],errs[1],errs[2],errs[3]);
		return false;
	}

	// size_t local[3];
	size_t global[3] = {num_map_output_CNN, height_image_output_CNN,width_image_output_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_kernel[FORWARD_OUT], 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_OUT. Error Code=%d\n", err); 
		return false;
	}

	clFinish(command_queue);
	// for (int i = 0; i < num_neuron_output_CNN; i++) {
	// 	neuron_output[i] = 0.0;
	// 	for (int c = 0; c < num_neuron_C5_CNN; c++) {
	// 		neuron_output[i] += weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
	// 	}
	// 	neuron_output[i] += bias_output[i];
	// 	neuron_output[i] = activation_function_tanh(neuron_output[i]);
	// }
	return true;
}





