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
/*
bool CNN::Backward_output()
{
	init_variable(delta_neuron_output, 0.0, num_neuron_output_CNN);
	for (int i = 0; i < num_neuron_output_CNN; i++) {
		delta_neuron_output[i] = activation_function_tanh_derivative(neuron_output[i])
				* loss_function_mse_derivative(neuron_output[i], data_single_label[i]);
	}
	return true;
}
*/
bool CNN::Backward_output()
{
        init_variable(delta_neuron_output, 0.0, num_neuron_output_CNN);

        float dE_dy[num_neuron_output_CNN];
        init_variable(dE_dy, 0.0, num_neuron_output_CNN);
        loss_function_gradient(neuron_output, data_single_label, dE_dy, num_neuron_output_CNN); // 损失函数: mean squared error(均方差)

        // delta = dE/da = (dE/dy) * (dy/da)
        for (int i = 0; i < num_neuron_output_CNN; i++) {
                float dy_da[num_neuron_output_CNN];
                init_variable(dy_da, 0.0, num_neuron_output_CNN);

                dy_da[i] = activation_function_tanh_derivative(neuron_output[i]);
                delta_neuron_output[i] = dot_product(dE_dy, dy_da, num_neuron_output_CNN);
        }
        return true;
}


bool CNN::Backward_C5()
{
	init_variable(delta_neuron_C5, 0.0, num_neuron_C5_CNN);
	init_variable(delta_weight_output, 0.0, len_weight_output_CNN);
	init_variable(delta_bias_output, 0.0, len_bias_output_CNN);

	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])
	// pre_w_kj_delta[j] += pre_delta[j]*n[k];
	// pre_b_kj_delta[j] += pre_delta[j]
	for (int j = 0; j < num_neuron_output_CNN; j++) {
		for (int k = 0; k < num_neuron_C5_CNN; k++) {
			int addr1 = k * num_neuron_output_CNN + j;    //当前权重
			int addr2 = j;
			delta_neuron_C5[k] += delta_neuron_output[j] * weight_output[addr1]
								  * activation_function_tanh_derivative(neuron_C5[k]);
			delta_weight_output[addr1] += delta_neuron_output[j] * neuron_C5[k];
			delta_bias_output[addr2] += delta_neuron_output[j];
		}
	}

	return true;
}

// hot path
bool CNN::Backward_S4()
{
	init_variable(delta_neuron_S4, 0.0, num_neuron_S4_CNN);
	init_variable(delta_weight_C5, 0.0, len_weight_C5_CNN);
	init_variable(delta_bias_C5, 0.0, len_bias_C5_CNN);

	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])
	// pre_w_kj_delta[j] += pre_delta[j]*n[k];
	// pre_b_kj_delta[j] += pre_delta[j]

	// 循环次数120 * 16 * 25
	for (int outc = 0; outc < num_map_C5_CNN; outc++) {
		//x, y为0，去除两层循环
		int index = (outc*height_image_C5_CNN*width_image_C5_CNN);  //C5 当前神经元 j
		for (int inc = 0; inc < num_map_S4_CNN; inc++) {
			int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc); //找到对应的卷积核
			int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入

			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				// for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr31 = addr1 + wy*width_kernel_conv_CNN ;  //卷积核索引 W_kj
					int addr32 = addr31 + 1;  //卷积核索引 W_kj
					int addr33 = addr31 + 2;  //卷积核索引 W_kj
					int addr34 = addr31 + 3;  //卷积核索引 W_kj
					int addr35 = addr31 + 4;  //卷积核索引 W_kj
					int addr41 = addr2 + wy*width_image_S4_CNN;     //S4中的像素索引 S4 k
					int addr42 = addr41 + 1;     //S4中的像素索引 S4 k
					int addr43 = addr41 + 2;     //S4中的像素索引 S4 k
					int addr44 = addr41 + 3;     //S4中的像素索引 S4 k
					int addr45 = addr41 + 4;     //S4中的像素索引 S4 k

					delta_neuron_S4[addr41] += delta_neuron_C5[index] * weight_C5[addr31]
											* activation_function_tanh_derivative(neuron_S4[addr41]);
					delta_neuron_S4[addr42] += delta_neuron_C5[index] * weight_C5[addr32]
											* activation_function_tanh_derivative(neuron_S4[addr42]);
					delta_neuron_S4[addr43] += delta_neuron_C5[index] * weight_C5[addr33]
											* activation_function_tanh_derivative(neuron_S4[addr43]);
					delta_neuron_S4[addr44] += delta_neuron_C5[index] * weight_C5[addr34]
											* activation_function_tanh_derivative(neuron_S4[addr44]);
					delta_neuron_S4[addr45] += delta_neuron_C5[index] * weight_C5[addr35]
											* activation_function_tanh_derivative(neuron_S4[addr45]);
					delta_weight_C5[addr31] += delta_neuron_C5[index] * neuron_S4[addr41];
					delta_weight_C5[addr32] += delta_neuron_C5[index] * neuron_S4[addr42];
					delta_weight_C5[addr33] += delta_neuron_C5[index] * neuron_S4[addr43];
					delta_weight_C5[addr34] += delta_neuron_C5[index] * neuron_S4[addr44];
					delta_weight_C5[addr35] += delta_neuron_C5[index] * neuron_S4[addr45];
				// }
			}

		}
		delta_bias_C5[outc] += delta_neuron_C5[index]*400;
	}

	return true;
}

bool CNN::Backward_C3()
{
	init_variable(delta_neuron_C3, 0.0, num_neuron_C3_CNN);
	init_variable(delta_weight_S4, 0.0, len_weight_S4_CNN);
	init_variable(delta_bias_S4, 0.0, len_bias_S4_CNN);
	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])*scale_factor
	// pre_w_kj_delta[j] += pre_delta[j]*n[k] * scale_factor;
	// pre_b_kj_delta[j] += pre_delta[j]
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	for (int outc=0; outc<num_map_S4_CNN; outc++) {
		int block = width_image_C3_CNN * height_image_C3_CNN * outc; //C3
		for (int y=0; y<height_image_S4_CNN; y++) {
			for (int x=0; x<width_image_S4_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (outc*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x; //S4 当前神经元j

				for (int m = 0; m < height_kernel_pooling_CNN; m++) {
					for (int n = 0; n < width_kernel_pooling_CNN; n++) {
						int addr1 = outc;  // 权重
					    int addr2 = block + (rows + m) * width_image_C3_CNN + cols + n; //C3 神经元 k
						int addr3 = outc;
						delta_neuron_C3[addr2] += delta_neuron_S4[index] * weight_S4[addr1]
											  * activation_function_tanh_derivative(neuron_C3[addr2]) * scale_factor;
						delta_weight_S4[addr1] += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
						delta_bias_S4[addr3] += delta_neuron_S4[index];
					}
				}
			}//index
		}
	}
	return true;
}

// hot path
bool CNN::Backward_S2()
{
	init_variable(delta_neuron_S2, 0.0, num_neuron_S2_CNN);
	init_variable(delta_weight_C3, 0.0, len_weight_C3_CNN);
	init_variable(delta_bias_C3, 0.0, len_bias_C3_CNN);

	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])
	// pre_w_kj_delta[j] += pre_delta[j]*n[k];
	// pre_b_kj_delta[j] += pre_delta[j]
	// 循环次数16 * 100 * 3 * 25
	for (int outc = 0; outc < num_map_C3_CNN; outc++) {
		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				int index = (outc*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //C3 当前神经元 j
				for (int inc = 0; inc < num_map_S2_CNN; inc++) {
					if (!tbl[inc][outc]) continue;
					int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S2_CNN * outc + inc); //找到对应的卷积核
					int addr2 = height_image_S2_CNN*width_image_S2_CNN*inc;   //找到对应的S2输入
					addr2 +=  y * width_image_S2_CNN + x;  //S2 k

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						// for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                            int addr31 = addr1 + wy*width_kernel_conv_CNN;  //卷积核索引 W_kj
                            int addr32 = addr31 + 1;  //卷积核索引 W_kj
                            int addr33 = addr31 + 2;  //卷积核索引 W_kj
                            int addr34 = addr31 + 3;  //卷积核索引 W_kj
                            int addr35 = addr31 + 4;  //卷积核索引 W_kj
                            int addr41 = addr2 + wy*width_image_S2_CNN;     //S2中的像素索引 S2 k
                            int addr42 = addr41 + 1;     //S2中的像素索引 S2 k
                            int addr43 = addr41 + 2;     //S2中的像素索引 S2 k
                            int addr44 = addr41 + 3;     //S2中的像素索引 S2 k
                            int addr45 = addr41 + 4;     //S2中的像素索引 S2 k
                            delta_neuron_S2[addr41] += delta_neuron_C3[index] * weight_C3[addr31]
                                                  * activation_function_tanh_derivative(neuron_S2[addr41]);
                            delta_neuron_S2[addr42] += delta_neuron_C3[index] * weight_C3[addr32]
                                                  * activation_function_tanh_derivative(neuron_S2[addr42]);
                            delta_neuron_S2[addr43] += delta_neuron_C3[index] * weight_C3[addr33]
                                                  * activation_function_tanh_derivative(neuron_S2[addr43]);
                            delta_neuron_S2[addr44] += delta_neuron_C3[index] * weight_C3[addr34]
                                                  * activation_function_tanh_derivative(neuron_S2[addr44]);
                            delta_neuron_S2[addr45] += delta_neuron_C3[index] * weight_C3[addr35]
                                                  * activation_function_tanh_derivative(neuron_S2[addr45]);
                            delta_weight_C3[addr31] += delta_neuron_C3[index] * neuron_S2[addr41];
                            delta_weight_C3[addr32] += delta_neuron_C3[index] * neuron_S2[addr42];
                            delta_weight_C3[addr33] += delta_neuron_C3[index] * neuron_S2[addr43];
                            delta_weight_C3[addr34] += delta_neuron_C3[index] * neuron_S2[addr44];
                            delta_weight_C3[addr35] += delta_neuron_C3[index] * neuron_S2[addr45];
						// }
					}
					delta_bias_C3[outc] += delta_neuron_C3[index]*25;
				}
			} //index
		}
	}

	return true;
}

bool CNN::Backward_C1()
{
	init_variable(delta_neuron_C1, 0.0, num_neuron_C1_CNN);
	init_variable(delta_weight_S2, 0.0, len_weight_S2_CNN);
	init_variable(delta_bias_S2, 0.0, len_bias_S2_CNN);
	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])*scale_factor
	// pre_w_kj_delta[j] += pre_delta[j]*n[k] * scale_factor;
	// pre_b_kj_delta[j] += pre_delta[j]
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	for (int outc=0; outc<num_map_S2_CNN; outc++) {
		int block = width_image_C1_CNN * height_image_C1_CNN * outc; //C1
		for (int y=0; y<height_image_S2_CNN; y++) {
			for (int x=0; x<width_image_S2_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (outc*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x; //S2 当前神经元j

				for (int m = 0; m < height_kernel_pooling_CNN; m++) {
					for (int n = 0; n < width_kernel_pooling_CNN; n++) {
						int addr1 = outc;  // 权重
						int addr2 = block + (rows + m) * width_image_C1_CNN + cols + n; //C1 神经元 k
						int addr3 = outc;
						delta_neuron_C1[addr2] += delta_neuron_S2[index] * weight_S2[addr1]
											  * activation_function_tanh_derivative(neuron_C1[addr2]) * scale_factor;
						delta_weight_S2[addr1] += delta_neuron_S2[index] * neuron_C1[addr2] * scale_factor;
						delta_bias_S2[addr3] += delta_neuron_S2[index];
					}
				}
			}//index
		}
	}

	return true;
}

// hot path
bool CNN::Backward_input()
{
	init_variable(delta_neuron_input, 0.0, num_neuron_input_CNN);
	init_variable(delta_weight_C1, 0.0, len_weight_C1_CNN);
	init_variable(delta_bias_C1, 0.0, len_bias_C1_CNN);

	// propagate delta to previous layer
	// cur_delta[k] += pre_delta[j] * W[kj]*ay(n[k])
	// pre_w_kj_delta[j] += pre_delta[j]*n[k];
	// pre_b_kj_delta[j] += pre_delta[j]
	// 循环次数 6*28*28*25
	for (int outc = 0; outc < num_map_C1_CNN; outc++) {
		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //C1 当前神经元 j
				for (int inc = 0; inc < num_map_input_CNN; inc++) {
					int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_input_CNN * outc + inc); //找到对应的卷积核
					int addr2 = height_image_input_CNN*width_image_input_CNN*inc;   //找到对应的input输入
					addr2 +=  y * width_image_input_CNN + x;  //input k

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                            int addr31 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                            int addr32 = addr31 + 1;  //卷积核索引 W_kj
                            int addr33 = addr31 + 2;  //卷积核索引 W_kj
                            int addr34 = addr31 + 3;  //卷积核索引 W_kj
                            int addr35 = addr31 + 4;  //卷积核索引 W_kj
                            int addr41 = addr2 + wy*width_image_input_CNN;     //input中的像素索引 input k
                            int addr42 = addr41 + 1;     //input中的像素索引 input k
                            int addr43 = addr41 + 2;     //input中的像素索引 input k
                            int addr44 = addr41 + 3;     //input中的像素索引 input k
                            int addr45 = addr41 + 4;     //input中的像素索引 input k
                            delta_neuron_input[addr41] += delta_neuron_C1[index] * weight_C1[addr31]
                                                  * activation_function_tanh_derivative(data_single_image[addr41]);
                            delta_neuron_input[addr42] += delta_neuron_C1[index] * weight_C1[addr32]
                                                  * activation_function_tanh_derivative(data_single_image[addr42]);
                            delta_neuron_input[addr43] += delta_neuron_C1[index] * weight_C1[addr33]
                                                  * activation_function_tanh_derivative(data_single_image[addr43]);
                            delta_neuron_input[addr44] += delta_neuron_C1[index] * weight_C1[addr34]
                                                  * activation_function_tanh_derivative(data_single_image[addr44]);
                            delta_neuron_input[addr45] += delta_neuron_C1[index] * weight_C1[addr35]
                                                  * activation_function_tanh_derivative(data_single_image[addr45]);
                            delta_weight_C1[addr31] += delta_neuron_C1[index] * data_single_image[addr41];
                            delta_weight_C1[addr32] += delta_neuron_C1[index] * data_single_image[addr42];
                            delta_weight_C1[addr33] += delta_neuron_C1[index] * data_single_image[addr43];
                            delta_weight_C1[addr34] += delta_neuron_C1[index] * data_single_image[addr44];
                            delta_weight_C1[addr35] += delta_neuron_C1[index] * data_single_image[addr45];
						}
					}
				}
				delta_bias_C1[outc] += delta_neuron_C1[index]*25;

			} //index
		}
	}

	return true;
}

