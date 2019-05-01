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


bool CNN::Forward_C1()
{
	cl_event evt[3];
	errs[0] = clEnqueueWriteBuffer(command_queue, Forward_C1_in, CL_FALSE, 0, num_neuron_input_CNN*sizeof(cl_float), data_single_image, 0, NULL, &evt[0]);
	errs[1] = clEnqueueWriteBuffer(command_queue, Forward_C1_bias, CL_FALSE, 0, len_bias_C1_CNN*sizeof(cl_float), bias_C1, 0, NULL, &evt[1]);
	errs[2] = clEnqueueWriteBuffer(command_queue, Forward_C1_weight, CL_FALSE, 0, len_weight_C1_CNN*sizeof(cl_float), weight_C1, 0, NULL, &evt[2]);
	// errs[3] = clEnqueueWriteBuffer(command_queue, Forward_C1_out, CL_FALSE, 0, num_neuron_C1_CNN*sizeof(cl_float), neuron_C1, 0, NULL, &evt[3]);
	if(errs[0]!= CL_SUCCESS || errs[1] != CL_SUCCESS || errs[2] != CL_SUCCESS ){
		cout << "can't write Forward_C1 buffer"<< endl;
		return false;
	}

	if (clSetKernelArg(Forward_C1_kernel, 0, sizeof(cl_mem), &Forward_C1_in) ||
		clSetKernelArg(Forward_C1_kernel, 1, sizeof(cl_mem), &Forward_C1_weight) ||
		clSetKernelArg(Forward_C1_kernel, 2, sizeof(cl_mem), &Forward_C1_bias) ||
		clSetKernelArg(Forward_C1_kernel, 3, sizeof(cl_mem), &Forward_C1_out) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_C1 arguments.");
		return false;
	}
	clWaitForEvents(3, evt);
	clReleaseEvent(evt[0]);
	clReleaseEvent(evt[1]);
	clReleaseEvent(evt[2]);
	// clReleaseEvent(evt[3]);

	// size_t local[3];
	size_t global[3] = {num_map_C1_CNN, height_image_C1_CNN,width_image_C1_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_C1_kernel, 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_C1. Error Code=%d\n", err); 
		return false;
	}

	errs[1] = clEnqueueReadBuffer(command_queue, Forward_C1_out, CL_TRUE,
							 0, num_neuron_C1_CNN*sizeof(cl_float), neuron_C1, 0, NULL, NULL);
	if (errs[1] != CL_SUCCESS)
	{
		printf("Error enqueuing read buffer command Forward_C1. Error Code=%d\n", errs[1]);
		return false;
	}

	// for (int channel = 0; channel < num_map_C1_CNN; channel++) {
	// 	for (int y = 0; y < height_image_C1_CNN; y++) {
	// 		for (int x = 0; x < width_image_C1_CNN; x++) {
	// 			int index = (channel*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //当前神经元
	// 			neuron_C1[index] = 0.0;
	// 			//卷积运算
	// 			for (int inc = 0; inc < num_map_input_CNN; inc++) {
	// 				int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN * num_map_input_CNN);
	// 				int addr2 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
	// 				const float* pw = &weight_C1[0] + addr1;       //卷积核
	// 				const float* pi = data_single_image + addr2;   //输入图像
	// 				float sum = 0.0;
	// 				const float* ppw = pw;
	// 				const float* ppi = pi + y * width_image_input_CNN + x;
	// 				for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
	// 					for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
	// 						sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
	// 					}
	// 				}
	// 				neuron_C1[index] += sum;
	// 			}
	// 			neuron_C1[index] += bias_C1[channel];     //加偏置
	// 			neuron_C1[index] = activation_function_tanh(neuron_C1[index]);  //激励函数
	// 		}
	// 	}
	// }
	return true;
}


bool CNN::Forward_S2()
{
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

	for (int i=0; i<num_map_S2_CNN; i++) {
		int block = width_image_C1_CNN * height_image_C1_CNN * i;
		for (int y=0; y<height_image_S2_CNN; y++) {
			for (int x=0; x<width_image_S2_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (i*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x;

                neuron_S2[index] = 0.0;
				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
                        neuron_S2[index] += weight_S2[i] * neuron_C1[(rows + m) * width_image_C1_CNN + cols + n + block];
					}
				}
				//
				neuron_S2[index] *= scale_factor;
				neuron_S2[index] += bias_S2[i] ;
				neuron_S2[index] = activation_function_tanh(neuron_S2[index]);
			}
		}
	}
	return true;
}

bool CNN::Forward_C3()
{
	cl_event evt[3];
	errs[0] = clEnqueueWriteBuffer(command_queue, Forward_C3_in, CL_FALSE, 0, num_neuron_S2_CNN*sizeof(cl_float), neuron_S2, 0, NULL, &evt[0]);
	errs[1] = clEnqueueWriteBuffer(command_queue, Forward_C3_bias, CL_FALSE, 0, len_bias_C3_CNN*sizeof(cl_float), bias_C3, 0, NULL, &evt[1]);
	errs[2] = clEnqueueWriteBuffer(command_queue, Forward_C3_weight, CL_FALSE, 0, len_weight_C3_CNN*sizeof(cl_float), weight_C3, 0, NULL, &evt[2]);
	// errs[3] = clEnqueueWriteBuffer(command_queue, Forward_C3_out, CL_FALSE, 0, num_neuron_C3_CNN*sizeof(cl_float), neuron_C3, 0, NULL, &evt[3]);
	if(errs[0]!= CL_SUCCESS || errs[1] != CL_SUCCESS || errs[2] != CL_SUCCESS ){
		cout << "can't write Forward_C3 buffer"<< endl;
		return false;
	}


	if (clSetKernelArg(Forward_C3_kernel, 0, sizeof(cl_mem), &Forward_C3_in) ||
		clSetKernelArg(Forward_C3_kernel, 1, sizeof(cl_mem), &Forward_C3_weight) ||
		clSetKernelArg(Forward_C3_kernel, 2, sizeof(cl_mem), &Forward_C3_bias) ||
		clSetKernelArg(Forward_C3_kernel, 3, sizeof(cl_mem), &Forward_C3_out) != CL_SUCCESS)
	{
		printf("Unable to set kernel Forward_C3 arguments.");
		return false;
	}
	clWaitForEvents(3, evt);
	clReleaseEvent(evt[0]);
	clReleaseEvent(evt[1]);
	clReleaseEvent(evt[2]);
	// clReleaseEvent(evt[3]);

	// size_t local[3];
	size_t global[3] = {num_map_C3_CNN, height_image_C3_CNN,width_image_C3_CNN};

	err = clEnqueueNDRangeKernel(command_queue, Forward_C3_kernel, 3, NULL, global, NULL /*local*/, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Unable to enqueue kernel command Forward_C3. Error Code=%d\n", err); 
		return false;
	}

	errs[1] = clEnqueueReadBuffer(command_queue, Forward_C3_out, CL_TRUE,
							 0, num_neuron_C3_CNN*sizeof(cl_float), neuron_C3, 0, NULL, NULL);
	if (errs[1] != CL_SUCCESS)
	{
		printf("Error enqueuing read buffer command Forward_C3. Error Code=%d\n", errs[1]);
		return false;
	}


	// for (int channel = 0; channel < num_map_C3_CNN; channel++) {
	// 	for (int y = 0; y < height_image_C3_CNN; y++) {
	// 		for (int x = 0; x < width_image_C3_CNN; x++) {
	// 			int index = (channel*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //当前神经元
	// 			neuron_C3[index] = 0.0;
	// 			//卷积运算
	// 			for (int inc = 0; inc < num_map_S2_CNN; inc++) {
	// 				if (!tbl[inc][channel]) continue;
	// 				int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C3_CNN * num_map_S2_CNN);
	// 				int addr2 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);  //输入图像
	// 				const float* pw = &weight_C3[0] + addr1;   //卷积核
	// 				const float* pi = &neuron_S2[0] + addr2;   //输入图像
	// 				float sum = 0.0;
	// 				const float* ppw = pw;
	// 				const float* ppi = pi + y * width_image_S2_CNN + x;
	// 				for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
	// 					for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
	// 						sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
	// 					}
	// 				}
	// 				neuron_C3[index] += sum;
	// 			}
	// 			neuron_C3[index] += bias_C3[channel];     //加偏置
	// 			neuron_C3[index] = activation_function_tanh(neuron_C3[index]);  //激励函数
	// 		}
	// 	}
	// }
	return true;
}

bool CNN::Forward_S4()
{
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	for (int i=0; i<num_map_S4_CNN; i++) {
		int block = width_image_C3_CNN * height_image_C3_CNN * i; //C3
		for (int y=0; y<height_image_S4_CNN; y++) {
			for (int x=0; x<width_image_S4_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (i*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x; //S4 当前神经元

                neuron_S4[index] = 0.0;
				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
                        neuron_S4[index] += weight_S4[i] * neuron_C3[(rows + m) * width_image_C3_CNN + cols + n + block];
					}
				}
				//
				neuron_S4[index] *= scale_factor;
				neuron_S4[index] += bias_S4[i] ;
				neuron_S4[index] = activation_function_tanh(neuron_S4[index]);
			}
		}
	}
	return true;
}

bool CNN::Forward_C5()
{
#if 1
	for (int channel = 0; channel < num_map_C5_CNN; channel++) {
		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				int index = (channel*height_image_C5_CNN*width_image_C5_CNN) + y*width_image_C5_CNN + x;  //当前神经元
				neuron_C5[index] = 0.0;
				//卷积运算
				for (int inc = 0; inc < num_map_S4_CNN; inc++) {
					int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C5_CNN * num_map_S4_CNN);
					int addr2 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
					const float* pw = &weight_C5[0] + addr1;       //卷积核
					const float* pi = &neuron_S4[0] + addr2;   //输入图像
					float sum = 0.0;
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_S4_CNN + x;
					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
						}
					}
					neuron_C5[index] += sum;
				}
				neuron_C5[index] += bias_C5[channel];     //加偏置
				neuron_C5[index] = activation_function_tanh(neuron_C5[index]);  //激励函数
			}
		}
	}
#else
	for (int channel = 0; channel < num_map_C5_CNN; channel++) {
		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				int index = (channel*height_image_C5_CNN*width_image_C5_CNN) + y*width_image_C5_CNN + x;  //C5 当前神经元
				for (int inc = 0; inc < num_map_S4_CNN; inc++) {
					int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * channel + inc); //找到对应的卷积核
					int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入
					addr2 += y * width_image_S4_CNN + x;
					//const float* pw = &weight_C5[0] + addr1;       //卷积核
					//const float* pi = &neuron_S4[0] + addr2;       //输入图像
					float sum = 0.0;
					//const float* ppw = pw;
					//const float* ppi = pi + y * width_image_S4_CNN + x;
					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                            int addr3 = wy*width_kernel_conv_CNN + wx;  //卷积核索引
                            int addr4 = wy*width_image_S4_CNN + wx;     //S4中的像素索引
                            sum += weight_C5[addr1 + addr3]*neuron_S4[addr2+addr4];
							//sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
						}
					}
					neuron_C5[index] += sum;
				}
				neuron_C5[index] += bias_C5[channel];     //加偏置
				neuron_C5[index] = activation_function_tanh(neuron_C5[index]);  //激励函数
			}
		}
	}
#endif
	return true;
}

bool CNN::Forward_output()
{
	for (int i = 0; i < num_neuron_output_CNN; i++) {
		neuron_output[i] = 0.0;
		for (int c = 0; c < num_neuron_C5_CNN; c++) {
			neuron_output[i] += weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
		}
		neuron_output[i] += bias_output[i];
		neuron_output[i] = activation_function_tanh(neuron_output[i]);
	}
	return true;
}





