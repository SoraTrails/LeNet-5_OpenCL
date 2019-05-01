#include "cnn.h"

using namespace std;

void CNN::init()
{
	//初始化数据
	int len1 = width_image_input_CNN * height_image_input_CNN * num_patterns_train_CNN;
	data_input_train = new float[len1];
	init_variable(data_input_train, -1.0, len1);

	int len2 = num_map_output_CNN * num_patterns_train_CNN;
	data_output_train = new float[len2];
	init_variable(data_output_train, -0.8, len2);

	int len3 = width_image_input_CNN * height_image_input_CNN * num_patterns_test_CNN;
	data_input_test = new float[len3];
	init_variable(data_input_test, -1.0, len3);

	int len4 = num_map_output_CNN * num_patterns_test_CNN;
	data_output_test = new float[len4];
	init_variable(data_output_test, -0.8, len4);

	std::fill(E_weight_C1, E_weight_C1 + len_weight_C1_CNN, 0.0);
	std::fill(E_bias_C1, E_bias_C1 + len_bias_C1_CNN, 0.0);
	std::fill(E_weight_S2, E_weight_S2 + len_weight_S2_CNN, 0.0);
	std::fill(E_bias_S2, E_bias_S2 + len_bias_S2_CNN, 0.0);
	std::fill(E_weight_C3, E_weight_C3 + len_weight_C3_CNN, 0.0);
	std::fill(E_bias_C3, E_bias_C3 + len_bias_C3_CNN, 0.0);
	std::fill(E_weight_S4, E_weight_S4 + len_weight_S4_CNN, 0.0);
	std::fill(E_bias_S4, E_bias_S4 + len_bias_S4_CNN, 0.0);
	E_weight_C5 = new float[len_weight_C5_CNN];
	std::fill(E_weight_C5, E_weight_C5 + len_weight_C5_CNN, 0.0);
	E_bias_C5 = new float[len_bias_C5_CNN];
	std::fill(E_bias_C5, E_bias_C5 + len_bias_C5_CNN, 0.0);
	E_weight_output = new float[len_weight_output_CNN];
	std::fill(E_weight_output, E_weight_output + len_weight_output_CNN, 0.0);
	E_bias_output = new float[len_bias_output_CNN];
	std::fill(E_bias_output, E_bias_output + len_bias_output_CNN, 0.0);

	//初始化Weight
	initWeightThreshold();
	//读取MNIST数据
	getSrcData();
}

int CNN::init_opencl(){
	std::cout << "get opencl ready" << std::endl;
	err = clGetPlatformIDs(1, &platform_id, &num_platforms_returned);
	if (err != CL_SUCCESS)
	{
		printf("Unable to get Platform ID. Error Code=%d\n", err);
		return -1;
	}
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devs_returned);
	if (err != CL_SUCCESS)
	{
		printf("Unable to get Device ID. Error Code=%d\n", err);
		return -1;
	}
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform_id;
	properties[2] = 0;
	//	create context
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create context. Error Code=%d\n", err);
		return -1;
	}
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create command queue. Error Code=%d\n", err);
		return -1;
	}
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src; // char string to hold kernel source
					  // initialize inputMatrix with some data and print it

	fp = fopen("../src/kernel.cl", "rb");
	if (fp == NULL){
		printf("error open src file\n");
		return -1;
	}

	fseek(fp, 0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = (char*)malloc(sizeof(char)*(filelen + 1));
	readlen = fread(kernel_src, 1, filelen, fp);
	if (readlen != filelen) {
		printf("error reading file\n");
		fclose(fp);
		return -1;
	}
	// ensure the string is	NULL terminated
	kernel_src[readlen] = '\0';
	fclose(fp);

	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create program object. Error Code=%d\n", err);
		free(kernel_src);
		return -1;
	}

	err = clBuildProgram(program, 0, NULL, "-DfilterSize=3", NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Build failed. Error Code=%d\n", err);
		size_t len = 0;
		cl_int ret = CL_SUCCESS;
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer = (char*)calloc(len, sizeof(char));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);	
		printf(" --- Build Log --- %d \n %s\n",ret, buffer);
		free(buffer);
		free(kernel_src);
		return -1;
	}
	free(kernel_src);

	Forward_C1_in = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   num_neuron_input_CNN*sizeof(cl_float), NULL, &errs[0]);
	Forward_C1_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
								   num_neuron_C1_CNN*sizeof(cl_float), NULL, &errs[1]);
	Forward_C1_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   len_bias_C1_CNN*sizeof(cl_float), NULL, &errs[2]);
	Forward_C1_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   len_weight_C1_CNN*sizeof(cl_float), NULL, &errs[3]);
	if(errs[0] != CL_SUCCESS ||errs[1] != CL_SUCCESS ||errs[2] != CL_SUCCESS ||errs[3] != CL_SUCCESS ){
		cout << "can't create Forward_C1 memory"<< endl;
		return -1;
	}

	Forward_C3_in = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   num_neuron_S2_CNN*sizeof(cl_float), NULL, &errs[0]);
	Forward_C3_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
								   num_neuron_C3_CNN*sizeof(cl_float), NULL, &errs[1]);
	Forward_C3_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   len_bias_C3_CNN*sizeof(cl_float), NULL, &errs[2]);
	Forward_C3_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								   len_weight_C3_CNN*sizeof(cl_float), NULL, &errs[3]);
	if(errs[0]!= CL_SUCCESS ||errs[1] != CL_SUCCESS ||errs[2] != CL_SUCCESS ||errs[3] != CL_SUCCESS ){
		cout << "can't create Forward_C3 memory"<< endl;
		return -1;
	}

	Forward_C1_kernel = clCreateKernel(program, "kernel_forward_c1", &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create kernel object Forward_C1_kernel. Error Code=%d\n", err); 
		return -1;
	}
	Forward_C3_kernel = clCreateKernel(program, "kernel_forward_c3", &err);
	if (err != CL_SUCCESS)
	{
		printf("Unable to create kernel object Forward_C3_kernel. Error Code=%d\n", err); 
		return -1;
	}
	std::cout << "opencl ready" << std::endl;
	return 0;
}

float CNN::uniform_rand(float min, float max)
{
	//std::mt19937 gen(1);
	std::random_device rd;
    std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dst(min, max);
	return dst(gen);
}

bool CNN::uniform_rand(float* src, int len, float min, float max)
{
	for (int i = 0; i < len; i++) {
		src[i] = uniform_rand(min, max);
	}
	return true;
}

bool CNN::initWeightThreshold()
{
	srand(time(0) + rand());
	const float scale = 6.0;

	float min_ = -std::sqrt(scale / (25.0 + 150.0));
	float max_ = std::sqrt(scale / (25.0 + 150.0));
	uniform_rand(weight_C1, len_weight_C1_CNN, min_, max_);
	for (int i = 0; i < len_bias_C1_CNN; i++) {
		bias_C1[i] = 0.0;
	}

	min_ = -std::sqrt(scale / (4.0 + 1.0));
	max_ = std::sqrt(scale / (4.0 + 1.0));
	uniform_rand(weight_S2, len_weight_S2_CNN, min_, max_);
	for (int i = 0; i < len_bias_S2_CNN; i++) {
		bias_S2[i] = 0.0;
	}

	min_ = -std::sqrt(scale / (150.0 + 400.0));
	max_ = std::sqrt(scale / (150.0 + 400.0));
	uniform_rand(weight_C3, len_weight_C3_CNN, min_, max_);
	for (int i = 0; i < len_bias_C3_CNN; i++) {
		bias_C3[i] = 0.0;
	}

	min_ = -std::sqrt(scale / (4.0 + 1.0));
	max_ = std::sqrt(scale / (4.0 + 1.0));
	uniform_rand(weight_S4, len_weight_S4_CNN, min_, max_);
	for (int i = 0; i < len_bias_S4_CNN; i++) {
		bias_S4[i] = 0.0;
	}

	min_ = -std::sqrt(scale / (400.0 + 3000.0));
	max_ = std::sqrt(scale / (400.0 + 3000.0));
	uniform_rand(weight_C5, len_weight_C5_CNN, min_, max_);
	for (int i = 0; i < len_bias_C5_CNN; i++) {
		bias_C5[i] = 0.0;
	}

	min_ = -std::sqrt(scale / (120.0 + 10.0));
	max_ = std::sqrt(scale / (120.0 + 10.0));
	uniform_rand(weight_output, len_weight_output_CNN, min_, max_);
	for (int i = 0; i < len_bias_output_CNN; i++) {
		bias_output[i] = 0.0;
	}

    return true;
}





