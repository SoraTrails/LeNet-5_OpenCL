/*
 * cnn.h
 *
 *  Created on: Apr 27, 2017
 *      Author: copper
 */

#ifndef CNN_H_
#define CNN_H_

#include <vector>
#include <unordered_map>
#include <assert.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <string>
#include <sys/time.h>
#include <cstring>
#include <sstream>
#include <CL/cl.h>

// 各层图像大小
#define width_image_input_CNN		32 //归一化图像宽
#define height_image_input_CNN		32 //归一化图像高
#define width_image_C1_CNN          28
#define height_image_C1_CNN		    28
#define width_image_S2_CNN		    14
#define height_image_S2_CNN		    14
#define width_image_C3_CNN		    10
#define height_image_C3_CNN		    10
#define width_image_S4_CNN		    5
#define height_image_S4_CNN		    5
#define width_image_C5_CNN		    1
#define height_image_C5_CNN		    1
#define width_image_output_CNN		1
#define height_image_output_CNN		1

// 卷积核大小
#define width_kernel_conv_CNN		5 //卷积核大小
#define height_kernel_conv_CNN		5
#define width_kernel_pooling_CNN	2
#define height_kernel_pooling_CNN	2
#define size_pooling_CNN		    2

// 特征图数量   feature maps
#define num_map_input_CNN		1 //输入层map个数
#define num_map_C1_CNN			6 //C1层map个数
#define num_map_S2_CNN			6 //S2层map个数
#define num_map_C3_CNN			16 //C3层map个数
#define num_map_S4_CNN			16 //S4层map个数
#define num_map_C5_CNN			120 //C5层map个数
#define num_map_output_CNN		10 //输出层map个数

// MNIST
#define num_patterns_train_CNN		60000  //60000 //训练模式对数(总数)
#define num_patterns_test_CNN		10000   //10000 //测试模式对数(总数)

// Train
#define num_epochs_CNN			    100   //最大迭代次数
#define accuracy_rate_CNN		    0.985 //要求达到的准确率
#define learning_rate_CNN		    0.01  //学习率
#define eps_CNN				        1e-8

//
#define len_weight_C1_CNN		150   //C1层权值数，5*5*6*1=150
#define len_bias_C1_CNN			6     //C1层阈值数，6
#define len_weight_S2_CNN		6     //S2层权值数,1*6=6
#define len_bias_S2_CNN			6     //S2层阈值数,6
#define len_weight_C3_CNN		2400  //C3层权值数，5*5*16*6=2400
#define len_bias_C3_CNN			16    //C3层阈值数,16
#define len_weight_S4_CNN		16    //S4层权值数，1*16=16
#define len_bias_S4_CNN			16    //S4层阈值数，16
#define len_weight_C5_CNN		48000 //C5层权值数，5*5*16*120=48000
#define len_bias_C5_CNN			120   //C5层阈值数，120
#define len_weight_output_CNN	1200  //输出层权值数，120*10=1200
#define len_bias_output_CNN		10    //输出层阈值数，10

#define num_neuron_input_CNN     1024 //输入层神经元数，32*32=1024
#define num_neuron_C1_CNN        4704 //C1层神经元数，28*28*6=4704
#define num_neuron_S2_CNN		 1176 //S2层神经元数，14*14*6=1176
#define num_neuron_C3_CNN		 1600 //C3层神经元数，10*10*16=1600
#define num_neuron_S4_CNN		 400  //S4层神经元数，5*5*16=400
#define num_neuron_C5_CNN		 120  //C5层神经元数，1*120=120
#define num_neuron_output_CNN    10   //输出层神经元数，1*10=10
// OpenCL 
#define FORWARD_NUM  6
#define FORWARD_C1   0
#define FORWARD_S2   1
#define FORWARD_C3   2
#define FORWARD_S4   3
#define FORWARD_C5   4
#define FORWARD_OUT  5

#define BACKWARD_NUM 7
#define BACKWARD_OUT 0
#define BACKWARD_C5  1
#define BACKWARD_S4  2
#define BACKWARD_C3  3
#define BACKWARD_S2  4
#define BACKWARD_C1  5
#define BACKWARD_IN  6
//
class CNN {
public:
	CNN();
	~CNN();

	void init(char* model=NULL);
	bool train();
	int  predict(const unsigned char *data, int width, int height);
	bool readModelFile(const char *name);
	bool  saveMiddlePic(int index);
	int init_opencl();

protected:
	void release();                         //释放申请的空间

	//init
	bool initWeightThreshold();             //初始化，产生[-1, 1]之间的随机小数
	void init_variable(float* val, float c, int len);
	bool uniform_rand(float* src, int len, float min, float max);
	float uniform_rand(float min, float max);

    //mnist
	int  reverseInt(int i);
	void readMnistImages(std::string filename, float* data_dst, int num_image);
	void readMnistLabels(std::string filename, float* data_dst, int num_image);
	bool getSrcData();                    //读取MNIST数据

	//math_functions
	float activation_function_tanh(float x); //激活函数:tanh
	float activation_function_tanh_derivative(float x); //激活函数tanh的导数
	float activation_function_identity(float x);
	float activation_function_identity_derivative(float x);
	float loss_function_mse(float y, float t); //损失函数:mean squared error
	float loss_function_mse_derivative(float y, float t);
	void loss_function_gradient(const float* y, const float* t, float* dst, int len);
	float dot_product(const float* s1, const float* s2, int len); //点乘
	bool muladd(const float* src, float c, int len, float* dst); //dst[i] += c * src[i]

	//model
	//bool reaModelFile(const char *name);
	bool saveModelFile(const char* name); //将训练好的model保存起来，包括各层的节点数，权值和阈值

	//
	int get_index(int x, int y, int channel, int width, int height, int depth);

	bool Forward_C1(int index, cl_mem & Forward_in_mem0); //前向传播
	bool Forward_S2();
	bool Forward_C3();
	bool Forward_S4();
	bool Forward_C5();
	bool Forward_output();

	bool Backward_output(int index);
	bool Backward_C5(); //反向传播
	bool Backward_S4();
	bool Backward_C3();
	bool Backward_S2();
	bool Backward_C1();
	bool Backward_input(int index);

	bool UpdateWeights(); //更新权值、阈值
	void update_weights_bias(const float* delta, float* e_weight, float* weight, int len);

	float test(); //训练完一次计算一次准确率

	//
	bool  bmp8(const float *data, int width, int height, const char *name);

private:
	float* data_input_train;   //原始标准输入数据，训练,范围：[-1, 1]
	float* data_output_train;  //原始标准期望结果，训练,取值：-0.8/0.8
	float* data_input_test;    //原始标准输入数据，测试,范围：[-1, 1]
	float* data_output_test;   //原始标准期望结果，测试,取值：-0.8/0.8
	float* data_single_image;
	float* data_single_label;

	float weight_C1[len_weight_C1_CNN];
	float bias_C1[len_bias_C1_CNN];
	float weight_S2[len_weight_S2_CNN];
	float bias_S2[len_bias_S2_CNN];
	float weight_C3[len_weight_C3_CNN];
	float bias_C3[len_bias_C3_CNN];
	float weight_S4[len_weight_S4_CNN];
	float bias_S4[len_bias_S4_CNN];
	float weight_C5[len_weight_C5_CNN];
	float bias_C5[len_bias_C5_CNN];
	float weight_output[len_weight_output_CNN];
	float bias_output[len_bias_output_CNN];

	float E_weight_C1[len_weight_C1_CNN];   //累积误差
	float E_bias_C1[len_bias_C1_CNN];
	float E_weight_S2[len_weight_S2_CNN];
	float E_bias_S2[len_bias_S2_CNN];
	float E_weight_C3[len_weight_C3_CNN];
	float E_bias_C3[len_bias_C3_CNN];
	float E_weight_S4[len_weight_S4_CNN];
	float E_bias_S4[len_bias_S4_CNN];
	float* E_weight_C5;
	float* E_bias_C5;
	float* E_weight_output;
	float* E_bias_output;

	float neuron_input[num_neuron_input_CNN]; //data_single_image
	float neuron_C1[num_neuron_C1_CNN];
	float neuron_S2[num_neuron_S2_CNN];
	float neuron_C3[num_neuron_C3_CNN];
	float neuron_S4[num_neuron_S4_CNN];
	float neuron_C5[num_neuron_C5_CNN];
	float neuron_output[num_neuron_output_CNN];

	float delta_neuron_output[num_neuron_output_CNN]; //神经元误差
	float delta_neuron_C5[num_neuron_C5_CNN];
	float delta_neuron_S4[num_neuron_S4_CNN];
	float delta_neuron_C3[num_neuron_C3_CNN];
	float delta_neuron_S2[num_neuron_S2_CNN];
	float delta_neuron_C1[num_neuron_C1_CNN];
	float delta_neuron_input[num_neuron_input_CNN];

	float delta_weight_C1[len_weight_C1_CNN]; //权值、阈值误差
	float delta_bias_C1[len_bias_C1_CNN];
	float delta_weight_S2[len_weight_S2_CNN];
	float delta_bias_S2[len_bias_S2_CNN];
	float delta_weight_C3[len_weight_C3_CNN];
	float delta_bias_C3[len_bias_C3_CNN];
	float delta_weight_S4[len_weight_S4_CNN];
	float delta_bias_S4[len_bias_S4_CNN];
	float delta_weight_C5[len_weight_C5_CNN];
	float delta_bias_C5[len_bias_C5_CNN];
	float delta_weight_output[len_weight_output_CNN];
	float delta_bias_output[len_bias_output_CNN];

	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_int errs[FORWARD_NUM+1];
	// cl_event events[FORWARD_NUM+1];
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;

	cl_kernel Forward_kernel[FORWARD_NUM];
	cl_kernel Backward_kernel[BACKWARD_NUM];
	cl_kernel Update_weights;

	cl_kernel Backward_kernel_s2_weight;
	cl_kernel Backward_kernel_s2_bias;
	cl_kernel Backward_kernel_input_weight;
	cl_kernel Backward_kernel_input_bias;

	cl_mem cl_data_input_train;
	cl_mem cl_label_input_train;
	cl_mem cl_data_input_test;
	cl_mem cl_label_input_test;

	//Forward
	cl_mem Forward_in_mem; // single_data_image
	cl_mem Forward_C1_mem;
	cl_mem Forward_S2_mem;
	cl_mem Forward_C3_mem;
	cl_mem Forward_S4_mem;
	cl_mem Forward_C5_mem;
	cl_mem Forward_out_mem;
	cl_mem Forward_bias[FORWARD_NUM];
	cl_mem Forward_weight[FORWARD_NUM];

	//Backward
	cl_mem Backward_bias[BACKWARD_NUM-1]; //delta
	cl_mem Backward_weight[BACKWARD_NUM-1];
	
	cl_mem Backward_out_mem; //neuron delta
	cl_mem Backward_C5_mem;
	cl_mem Backward_S4_mem;
	cl_mem Backward_C3_mem;
	cl_mem Backward_S2_mem;
	cl_mem Backward_C1_mem;
	cl_mem Backward_in_mem;

	//updateweights
	cl_mem Update_bias[FORWARD_NUM]; // e_delta
	cl_mem Update_weight[FORWARD_NUM];

	const int for_mem_bw_len[FORWARD_NUM][2]={
		{len_bias_C1_CNN,len_weight_C1_CNN},
		{len_bias_S2_CNN,len_weight_S2_CNN},
		{len_bias_C3_CNN,len_weight_C3_CNN},
		{len_bias_S4_CNN,len_weight_S4_CNN},
		{len_bias_C5_CNN,len_weight_C5_CNN},
		{len_bias_output_CNN,len_weight_output_CNN}
	};
	const int back_mem_bw_len[BACKWARD_NUM-1][2]={
		{len_bias_output_CNN,len_weight_output_CNN},
		{len_bias_C5_CNN,len_weight_C5_CNN},
		{len_bias_S4_CNN,len_weight_S4_CNN},
		{len_bias_C3_CNN,len_weight_C3_CNN},
		{len_bias_S2_CNN,len_weight_S2_CNN},
		{len_bias_C1_CNN,len_weight_C1_CNN}
	};
	const int for_mem_in_out_len[FORWARD_NUM+1]={
		num_neuron_input_CNN,num_neuron_C1_CNN,num_neuron_S2_CNN,
		num_neuron_C3_CNN,num_neuron_S4_CNN,num_neuron_C5_CNN,num_neuron_output_CNN
	};
	const int back_mem_in_out_len[BACKWARD_NUM]={
		num_neuron_output_CNN,num_neuron_C5_CNN,num_neuron_S4_CNN,
		num_neuron_C3_CNN,num_neuron_S2_CNN,num_neuron_C1_CNN,num_neuron_input_CNN
	};

	cl_mem *for_mem[FORWARD_NUM+1] = {&Forward_in_mem,&Forward_C1_mem,&Forward_S2_mem,&Forward_C3_mem,&Forward_S4_mem,&Forward_C5_mem,&Forward_out_mem};
	float *for_mem_src[FORWARD_NUM+1] = {neuron_input,neuron_C1,neuron_S2,neuron_C3,neuron_S4,neuron_C5,neuron_output};
	cl_mem *back_mem[BACKWARD_NUM] = {&Backward_out_mem,&Backward_C5_mem,&Backward_S4_mem,&Backward_C3_mem,&Backward_S2_mem,&Backward_C1_mem,&Backward_in_mem};
	float *back_mem_src[BACKWARD_NUM] = {delta_neuron_output,delta_neuron_C5,delta_neuron_S4,delta_neuron_C3,delta_neuron_S2,delta_neuron_C1,delta_neuron_input};

	std::string forward_kernel_name[FORWARD_NUM]={
		{"kernel_forward_c1"},
		{"kernel_forward_s2"},
		{"kernel_forward_c3"},
		{"kernel_forward_s4"},
		{"kernel_forward_c5"},
		{"kernel_forward_output"}
	};
	std::string backward_kernel_name[BACKWARD_NUM]={
		{"kernel_backward_output"},
		{"kernel_backward_c5"},
		{"kernel_backward_s4"},
		{"kernel_backward_c3"},
		{"kernel_backward_s2"},
		{"kernel_backward_c1"},
		{"kernel_backward_input"}
	};

	float *biases[FORWARD_NUM] = {bias_C1, bias_S2, bias_C3, bias_S4, bias_C5, bias_output};
	float *weights[FORWARD_NUM] = {weight_C1, weight_S2, weight_C3, weight_S4, weight_C5, weight_output};
};

#endif /* CNN_H_ */
