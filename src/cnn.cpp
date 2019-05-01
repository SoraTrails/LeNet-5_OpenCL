#include "cnn.h"

using namespace std;


CNN::CNN()
{
	data_input_train = NULL;
	data_output_train = NULL;
	data_input_test = NULL;
	data_output_test = NULL;
    data_single_image = NULL;
    data_single_label = NULL;
	E_weight_C5 = NULL;
	E_bias_C5 = NULL;
	E_weight_output = NULL;
    E_bias_output = NULL;
    std::cout << "Create" << std::endl;
}

CNN::~CNN()
{
	release();
}


void CNN::release()
{
	if (data_input_train) {
		delete[] data_input_train;
		data_input_train = NULL;
	}
	if (data_output_train) {
		delete[] data_output_train;
		data_output_train = NULL;
	}
	if (data_input_test) {
		delete[] data_input_test;
		data_input_test = NULL;
	}
	if (data_output_test) {
		delete[] data_output_test;
		data_output_test = NULL;
	}

	if (E_weight_C5) {
		delete[] E_weight_C5;
		E_weight_C5 = NULL;
	}
	if (E_bias_C5) {
		delete[] E_bias_C5;
		E_bias_C5 = NULL;
	}
	if (E_weight_output) {
		delete[] E_weight_output;
		E_weight_output = NULL;
	}
	if (E_bias_output) {
		delete[] E_bias_output;
		E_bias_output = NULL;
	}

	
	clReleaseDevice(device_id);
	clReleaseMemObject(Forward_C1_in);
	clReleaseMemObject(Forward_C1_out);
	clReleaseMemObject(Forward_C1_bias);
	clReleaseMemObject(Forward_C1_weight);
	clReleaseMemObject(Forward_C3_in);
	clReleaseMemObject(Forward_C3_out);
	clReleaseMemObject(Forward_C3_bias);
	clReleaseMemObject(Forward_C3_weight);
	clReleaseProgram(program);
	clReleaseKernel(Forward_C1_kernel);
	clReleaseKernel(Forward_C3_kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}



