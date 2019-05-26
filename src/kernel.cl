__constant int tbl[6][16] = {
	{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
	{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
	{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
	{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
	{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
	{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
};

__kernel void  kernel_forward_c1(__global float *in,
                      __constant float  *weight,
                      __global float  *bias,
                      __global float  *out,
					  int input_index)
{
	// printf("%d\n",input_index);
    //[6,28,28]
    //[1,28,28]
	int channel = get_global_id(0);
	int out_height = get_global_size(1);//28
	int out_width = get_global_size(2);//28
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 32;
	int in_height = 32;
    int index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	float out_val = 0.0;
    // __local float local_weight[32*32];
    // local_weight[x*32+y] = weight[channel * 25 + x*32+y];


    __constant const float* ppw = weight + channel * kernel_height * kernel_width;   //卷积核
    sum = 0.0;
    __global const float* ppi = in + input_index; ;
    for(int wy = 0; wy < kernel_height; wy++)  {
        for(int wx = 0; wx < kernel_width; wx++) {
            // sum += ppw[wy*kernel_width+wx] * ppi[y * in_width + x + wy * in_width + wx];
            sum += ppw[wy*kernel_width+wx] * ppi[(y+wy) * in_width + (x + wx)];
        }
    }
    out_val += sum;

	out_val += bias[channel];
	out_val = tanh((float)(out_val));
	out[index] = out_val;
}

__kernel void  kernel_forward_s2(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int out_height = get_global_size(1);
	int out_width = get_global_size(2);
	int kernel_width=2;
	int kernel_height=2;
	int in_width=28;
	int in_height=28;
	//TODO
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int out_height = get_global_size(1);
	int out_width = get_global_size(2);
    int y = get_global_id(1);
    int x = get_global_id(2);
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 14;
	int in_height = 14;
	int in_num = 6;
    int index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	float out_index = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out_index += sum;
	}
	out_index += bias[channel];
	out_index = tanh((float)(out_index));
	out[index] = out_index;
}

__kernel void  kernel_forward_s4(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int out_height = get_global_size(1);
	int out_width = get_global_size(2);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width=2;
	int kernel_height=2;
	int in_width=10;
	int in_height=10;
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int out_height=1;
	int out_width=1;
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 5;
	int in_height = 5;
	int in_num=16;

	int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_output(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int num_neuron_output_CNN=10;
	int in_num=120;
	out[channel] = 0.0;
	for (int c = 0; c < in_num; c++) {
		out[channel] += weight[c * num_neuron_output_CNN + channel] * in[c];
	}
	out[channel] += bias[channel];
	out[channel] = tanh((float)(out[channel]));

}

__kernel void  kernel_backward_output(
	__global float *in, //neuron_output
	__global float *label, //data_single_label
	__global float *out, //delta_neuron_output
	int index //index of label
)
{
	//[10]
	int channel = get_global_id(0);
	__global float *labels = label + index;

	out[channel] = 0.0f;
	const int num_neuron_output_CNN = 10;

	//can use local memory optimization
	float dE_dy[10];
	for(int i=0;i<num_neuron_output_CNN;i++){
		dE_dy[i] = in[i] - labels[i];
	}

	float dy_da[10];
	for(int i=0;i<num_neuron_output_CNN;i++){
		dy_da[i] = 0.0f;
	}
	dy_da[channel] = 1.0 - in[channel] * in[channel];
	float res = 0.0f;
	for(int i=0;i<num_neuron_output_CNN;i++){
		res += dy_da[i] * dE_dy[i];
	}
	out[channel] = res;
}

__kernel void  kernel_backward_c5(
	__global float *in, //delta_neuron_output
	__global float *neuron_C5, //neuron_C5(in)
	__global float *weight_output, //weight_output(in) 
	__global float *delta_weight, // delta_weight_output
	__global float *delta_bias,	 // delta_bias_output
	__global float *out //delta_neuron_C5
)
{
	//[120]
	int channel = get_global_id(0);
	int num_neuron_output_CNN = 10;
	out[channel] = 0.0;
	for (int j = 0; j < num_neuron_output_CNN; j++) {
		int addr1 = channel * num_neuron_output_CNN + j;    //当前权重
		out[channel] += in[j] * weight_output[addr1] * (1.0-neuron_C5[channel]*neuron_C5[channel]);
		delta_weight[addr1] = in[j] * neuron_C5[channel];
		// delta_bias[j] += in[j];
	}
	if(channel < 10){
		delta_bias[channel] = 120*in[channel];
	}
}

__kernel void  kernel_backward_s4(
	__global float *in, //delta_neuron_C5
	__global float *neuron_S4, //neuron_S4(in)
	__global float *weight_C5, //weight_C5(in) 
	__global float *delta_weight, // delta_weight_C5
	__global float *delta_bias,	 // delta_bias_C5
	__global float *out //delta_neuron_S4
){
	//[16,120]
	//[1,120]
	int inc = get_global_id(0);
	int outc = get_global_id(1);
	const int width_kernel_conv_CNN = 5;
	const int height_kernel_conv_CNN = 5;
	const int num_map_S4_CNN = 16;
	const int height_image_S4_CNN = 5;
	const int width_image_S4_CNN = 5;
	const int num_map_C5_CNN = 120;
	__local float out_tmp[120][25];
	for(int i=0;i<25;i++)
		out_tmp[outc][i] = 0;

	// for (int outc = 0; outc < num_map_C5_CNN; outc++) {
	int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc); //找到对应的卷积核
	int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入

	for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
		for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
			int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
			int addr4 = addr2 + wy*width_image_S4_CNN + wx;     //S4中的像素索引 S4 k
			int addr4_tmp = wy*width_image_S4_CNN + wx;
			out_tmp[outc][addr4_tmp] += in[outc] * weight_C5[addr3] * (1.0 - neuron_S4[addr4] * neuron_S4[addr4]);
			delta_weight[addr3] = in[outc] * neuron_S4[addr4];
			// delta_bias[outc] += in[outc];
		}
	}
	if(inc == 0)
		delta_bias[outc] = in[outc] * 25;
	if(outc < 60){
		for(int i=0;i<25;i++)
			out_tmp[outc][i] += out_tmp[outc + 60][i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(outc < 30){
		for(int i=0;i<25;i++)
			out_tmp[outc][i] += out_tmp[outc + 30][i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(outc < 15){
		for(int i=0;i<25;i++)
			out_tmp[outc][i] += out_tmp[outc + 15][i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(outc < 5){
		for(int i=0;i<25;i++)
			out_tmp[outc][i] += out_tmp[outc + 5][i] + out_tmp[outc + 10][i] ;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(outc == 0){
		for(int i=0;i<25;i++)
			out[height_image_S4_CNN*width_image_S4_CNN*inc + i] = out_tmp[0][i] + out_tmp[1][i] + out_tmp[2][i] + out_tmp[3][i] + out_tmp[4][i];
	}
}

__kernel void  kernel_backward_c3(
	__global float *in, //delta_neuron_S4
	__global float *neuron_C3, //neuron_C3(in)
	__global float *weight_S4, //weight_S4(in) 
	__global float *delta_weight, // delta_weight_S4
	__global float *delta_bias,	 // delta_bias_S4
	__global float *out //delta_neuron_C3
){
	//[16]
	int outc = get_global_id(0);
	const float scale_factor = 0.25f;
	const int width_kernel_pooling_CNN = 2;
	const int height_kernel_pooling_CNN = 2;
	const int width_image_C3_CNN = 10;
	const int height_image_C3_CNN = 10;
	const int width_image_S4_CNN = 5;
	const int height_image_S4_CNN = 5;
	int block = width_image_C3_CNN * height_image_C3_CNN * outc; //C3
	
	delta_weight[outc] = 0.0f;
	delta_bias[outc] = 0.0f;
	for(int i=0;i<width_image_C3_CNN * height_image_C3_CNN;i++){
		out[outc * 100 + i] = 0.0f;
	}

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
					out[addr2] += in[index] * weight_S4[addr1] * (1.0 - neuron_C3[addr2] * neuron_C3[addr2]) * scale_factor;
					delta_weight[addr1] += in[index] * neuron_C3[addr2] * scale_factor;
					delta_bias[addr3] += in[index];
				}
			}
		}//index
	}

}

__kernel void  kernel_backward_s2(
	__global float *in, //delta_neuron_C3
	__global float *neuron_S2, //neuron_S2(in)
	__global float *weight_C3, //weight_C3(in) 
	__global float *delta_weight, // delta_weight_C3
	__global float *delta_bias,	 // delta_bias_C3
	__global float *out //delta_neuron_S2
){
	//[6]
	int inc = get_global_id(0);
	const int num_map_C3_CNN = 16;
	const int height_image_C3_CNN = 10;
	const int width_image_C3_CNN = 10;
	const int width_kernel_conv_CNN = 5;
	const int height_kernel_conv_CNN = 5;
	const int num_map_S2_CNN = 6;
	const int height_image_S2_CNN = 14;
	const int width_image_S2_CNN = 14;

	//init
	for(int i=0;i<196;i++){
		out[inc * 196 + i] = 0.0f;
	}
	for(int i=0;i<400;i++){
		delta_weight[inc * 400 + i] = 0.0f;
	}
	
	__local float delta_bias_local[6][16];

	for(int i=0;i<num_map_C3_CNN;i++){
		delta_bias_local[inc][i] = 0.0f;
	}

	for (int outc = 0; outc < num_map_C3_CNN; outc++) {
		if (!tbl[inc][outc]) continue;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				int index = (outc*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //C3 当前神经元 j
				////////////
				// inc loop 
				int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S2_CNN * outc + inc); //找到对应的卷积核
				int addr2 = height_image_S2_CNN*width_image_S2_CNN*inc;   //找到对应的S2输入
				addr2 += y * width_image_S2_CNN + x;  //S2 k
				for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
					for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
						int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
						int addr4 = addr2 + wy*width_image_S2_CNN + wx;     //S2中的像素索引 S2 k
						int addr5 = outc;
						out[addr4] += in[index] * weight_C3[addr3] * (1.0 - neuron_S2[addr4] * neuron_S2[addr4]);
						delta_weight[addr3] += in[index] * neuron_S2[addr4];
						// delta_bias[addr5] += in[index];
					}
				}
				//TODO bias
				delta_bias_local[inc][outc] += in[index] * 25;
				////////////
			} //index
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(inc < 3){
		for(int i=0;i<num_map_C3_CNN;i++){
			delta_bias_local[inc][i] += delta_bias_local[inc+3][i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(inc == 0){
		for(int i=0;i<num_map_C3_CNN;i++){
			delta_bias[i] = delta_bias_local[inc][i] + delta_bias_local[inc+1][i] + delta_bias_local[inc+2][i];
		}
	}
}


__kernel void  kernel_backward_c1(
	__global float *in, //delta_neuron_S2
	__global float *neuron_C1, //neuron_C1(in)
	__global float *weight_S2, //weight_S2(in) 
	__global float *delta_weight, // delta_weight_S2
	__global float *delta_bias,	 // delta_bias_S2
	__global float *out //delta_neuron_C1
){
	//[6]
	int outc = get_global_id(0);
	const float scale_factor = 0.25f;
	const int width_kernel_pooling_CNN = 2;
	const int height_kernel_pooling_CNN = 2;
	const int width_image_C1_CNN = 28;
	const int height_image_C1_CNN = 28;
	const int width_image_S2_CNN = 14;
	const int height_image_S2_CNN = 14;
	delta_weight[outc] = 0.0f;
	delta_bias[outc] = 0.0f;
	for(int i=0;i<width_image_C1_CNN * height_image_C1_CNN;i++){
		out[outc * 28 * 28 + i] = 0.0f;
	}
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
					out[addr2] += in[index] * weight_S2[addr1] * (1.0 - neuron_C1[addr2]*neuron_C1[addr2]) * scale_factor;
					delta_weight[addr1] += in[index] * neuron_C1[addr2] * scale_factor;
					delta_bias[addr3] += in[index];
				}
			}
		}//index
	}

}
__kernel void  kernel_backward_input(
	__global float *in, //delta_neuron_C1
	__global float *neuron_input, //data_single_image(in)
	__global float *weight_C1, //weight_C1(in) 
	__global float *delta_weight, // delta_weight_C1
	__global float *delta_bias,	 // delta_bias_C1
	__global float *out, //delta_neuron_input
	int index // index of data_single_image
){
	//TODO
	//[1]
	// int wx = get_global_id(0);
	// int wy = get_global_id(1);
	__global float *data_single_image = neuron_input + index;
	int width_image_input_CNN = 32;
	int height_image_input_CNN = 32;
	int width_image_C1_CNN = 28;
	int height_image_C1_CNN = 28;
	int width_kernel_conv_CNN = 5;
	int height_kernel_conv_CNN = 5;
	int num_map_C1_CNN = 6;
	for(int i=0;i<width_image_input_CNN*height_image_input_CNN;i++){
		out[i] = 0.0f;
	}
	for(int i=0;i<150;i++){
		delta_weight[i] = 0.0f;
	}
	for(int i=0;i<6;i++){
		delta_bias[i] = 0.0f;
	}

	for (int outc = 0; outc < num_map_C1_CNN; outc++) {
		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //C1 当前神经元 j
				int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(outc); //找到对应的卷积核
				int addr2 = y * width_image_input_CNN + x;  //input k

				for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
					for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
						int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
						int addr4 = addr2 + wy*width_image_input_CNN + wx;     //input中的像素索引 input k
						int addr5 = outc;
						out[addr4] += in[index] * weight_C1[addr3] * (1.0 - data_single_image[addr4] * data_single_image[addr4]);
						delta_weight[addr3] += in[index] * data_single_image[addr4];
						delta_bias[addr5] += in[index];
					}
				}
			} //index
		}
	}
}

__kernel void kernel_update_weights(
	__global float * delta,
	__global float * e_weight,
	__global float * weight
){
	int i = get_global_id(0);
	e_weight[i] += delta[i] * delta[i];
	weight[i] -= 0.01 * delta[i] / (sqrt(e_weight[i]) + 1e-8);
}
