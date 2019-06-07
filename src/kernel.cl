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
    //[1,7,7]
	//-DfilterSize=5 -DBlockSize=7

	int channel = get_global_id(0);
	int out_height = 28;
	int out_width = 28;
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width = 5;
	// printf("0:%d %d %d\n", channel, y, x);
	int kernel_height = 5;
	int in_width = 32;
	int in_height = 32;
	int in_num = 1;
    int index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	float out_val = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__constant const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + input_index + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out_val += sum;
	}
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
	int out_height = 14;
	int out_width = 14;
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

	float out_index=0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out_index += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out_index *= 0.25;  //scale_factor;
	out_index += bias[channel] ;
	out[index] = tanh((float)(out_index));
}

__kernel void  kernel_forward_c3(__global float *in,
                      __constant float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	//[16,10,10]
	//[1,10,10]
	int channel = get_global_id(0);
	int out_height = 10;
	int out_width = 10;
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
		if (!tbl[inc][channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__constant const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
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
	int out_height = 5;
	int out_width = 5;
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

	float out_index = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out_index += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out_index *= 0.25;  //scale_factor;
	out_index += bias[channel] ;
	out[index] = tanh((float)(out_index));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
    // int  y = get_global_id(1);
    // int  x = get_global_id(2);
	int out_height=1;
	int out_width=1;
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 5;
	int in_height = 5;
	int in_num=16;

	int  index = channel*out_height*out_width;
	// int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	float out_index=0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out_index += sum;
	}
	out_index += bias[channel];
	out[index] = tanh((float)(out_index));
}

__kernel void  kernel_forward_output(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int num_neuron_output_CNN=10;
	int in_num=120;
	// out[channel] = 0.0;
	float out_channel = 0.0;
	for (int c = 0; c < in_num; c++) {
		out_channel += weight[c * num_neuron_output_CNN + channel] * in[c];
	}
	out_channel += bias[channel];
	out[channel] = tanh((float)(out_channel));

}

__kernel void  kernel_backward_output(
	__global float *in, //neuron_output
	__global float *label, //data_single_label
	__global float *out, //delta_neuron_output
	int index //index of label
)
{
	//[10]
	int i = get_global_id(0);
	__global float *labels = label + index;
	const int num_neuron_output_CNN = 10;

	float res = (in[i] - labels[i]) * (1.0 - in[i] * in[i]);

	out[i] = res;
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
	float out_channel = 0.0;
	for (int j = 0; j < num_neuron_output_CNN; j++) {
		int addr1 = channel * num_neuron_output_CNN + j;    //当前权重
		out_channel += in[j] * weight_output[addr1] * (1.0-neuron_C5[channel]*neuron_C5[channel]);
		delta_weight[addr1] = in[j] * neuron_C5[channel];
		// delta_bias[j] += in[j];
	}
	out[channel] = out_channel;
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
	//[16,5,5]
	//[1,5,5]
	int inc = get_global_id(0);
	int wy = get_global_id(1);
	int wx = get_global_id(2);
	const int width_kernel_conv_CNN = 5;
	const int height_kernel_conv_CNN = 5;
	const int num_map_S4_CNN = 16;
	const int height_image_S4_CNN = 5;
	const int width_image_S4_CNN = 5;
	const int num_map_C5_CNN = 120;
	int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入
	int addr4 = addr2 + wy*width_image_S4_CNN + wx;     //S4中的像素索引 S4 k
	float out_addr4=0;
	float neuron_S4_addr4 = neuron_S4[addr4];
	for (int outc = 0; outc < num_map_C5_CNN; outc++) {
		int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc); //找到对应的卷积核

		int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
		out_addr4 += in[outc] * weight_C5[addr3] * (1.0 - neuron_S4_addr4 * neuron_S4_addr4);
		delta_weight[addr3] = in[outc] * neuron_S4_addr4;
		// delta_bias[outc] += in[outc];
		if(inc == 0 && wx == 0 && wy == 0)
			delta_bias[outc] = in[outc]*400;
	}
	out[addr4] = out_addr4;
}

__kernel void  kernel_backward_c3(
	__global float *in, //delta_neuron_S4
	__global float *neuron_C3, //neuron_C3(in)
	__global float *weight_S4, //weight_S4(in) 
	__global float *delta_weight, // delta_weight_S4
	__global float *delta_bias,	 // delta_bias_S4
	__global float *out //delta_neuron_C3
){
	//[16,5,5]
	//[1,5,5]
	int outc = get_global_id(0);
	int y = get_global_id(1);
	int x = get_global_id(2);
	const float scale_factor = 0.25f;
	int block = 10 * 10 * outc; //C3
	int index = (outc*5*5) + y*5 + x; //S4 当前神经元j
	__local float w_tmp[5][5];
	__local float b_tmp[5][5];
	w_tmp[y][x] = 0;
	// delta_weight[outc] = 0.0f;
	// delta_bias[outc] = 0.0f;
	// for(int i=0;i<10 * 10;i++){
	// 	out[outc * 100 + i] = 0.0f;
	// }
	for (int m = 0; m < 2; m++) {
		for (int n = 0; n < 2; n++) {
			int addr2 = block + (y * 2 + m) * 10 + x * 2 + n; //C3 神经元 k
			out[addr2] = in[index] * weight_S4[outc] * (1.0 - neuron_C3[addr2] * neuron_C3[addr2]) * scale_factor;
			w_tmp[y][x] += in[index] * neuron_C3[addr2] * scale_factor;
		}
	}
	b_tmp[y][x] = in[index]*4;
	barrier(CLK_LOCAL_MEM_FENCE);
	if(x == 0 && y == 0){
		float tmpb=0,tmpw=0;
		for(int yy = 0;yy < 5;yy++)
			for(int xx = 0;xx < 5;xx++){
				tmpb += b_tmp[yy][xx];
				tmpw += w_tmp[yy][xx];
			}
		delta_weight[outc] = tmpw;
		delta_bias[outc] = tmpb;
	}
}

__kernel void  kernel_backward_s2(
	__global float *in, //delta_neuron_C3
	__global float *neuron_S2, //neuron_S2(in)
	__global float *weight_C3, //weight_C3(in) 
	// __global float *delta_weight, // delta_weight_C3
	// __global float *delta_bias,	 // delta_bias_C3
	__global float *out //delta_neuron_S2
){
	//[14,14,6]
	int yy = get_global_id(0);
	int xx = get_global_id(1);
	int inc = get_global_id(2);
	int addr4 = 14*14*inc+yy*14+xx;
	float out_addr4 = 0;
	float neuron_S2_addr4 = neuron_S2[addr4];
	for (int outc = 0; outc < 16; outc++) {
		if (!tbl[inc][outc]) continue;
		int addr1 = 5*5*(6 * outc + inc); //找到对应的卷积核
		for(int y = max(yy-4,0);y<=min(yy,9);y++){
			int wy = yy - y;
			for(int x = max(xx-4,0);x<=min(xx,9);x++){
				int wx = xx - x;
				int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
				int addr3 = addr1 + wy*5 + wx;  //卷积核索引 W_kj
				out_addr4 += in[index] * weight_C3[addr3] * (1-neuron_S2_addr4*neuron_S2_addr4);
			}
		}
	}
	out[addr4] = out_addr4;
}

__kernel void  kernel_backward_s2_weight(
	__global float *in, //delta_neuron_C3
	__global float *neuron_S2, //neuron_S2(in)
	// __global float *weight_C3 //weight_C3(in) 
	__global float *delta_weight // delta_weight_C3
	// __global float *delta_bias,	 // delta_bias_C3
	// __global float *out //delta_neuron_S2
){
	//[25,16,6]
	int wxy = get_global_id(0);
	int outc = get_global_id(1);
	int inc = get_global_id(2);
	int wy = wxy / 5;
	int wx = wxy - wy * 5;
	if (!tbl[inc][outc]) 
		return;
	int addr3 = 5*5*(6 * outc + inc) + wxy;
	float delta_weight_addr3 = 0;

	for (int y = 0; y < 10; y++) {
		for (int x = 0; x < 10; x++) {
			int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
			int addr2 = 14*14*inc +  y * 14 + x + wy*14 + wx;   //找到对应的S2输入
			delta_weight_addr3 += in[index] * neuron_S2[addr2];
		}
	}
	delta_weight[addr3] = delta_weight_addr3;
}

__kernel void  kernel_backward_s2_bias(
	__global float *in, //delta_neuron_C3
	// __global float *neuron_S2, //neuron_S2(in)
	// __global float *weight_C3 //weight_C3(in) 
	// __global float *delta_weight // delta_weight_C3
	__global float *delta_bias	 // delta_bias_C3
	// __global float *out //delta_neuron_S2
){
	//[16]
	int outc = get_global_id(0);
	float delta_bias_outc = 0;

	for (int inc = 0; inc < 6; inc++) {
		if (!tbl[inc][outc]) continue;
		for (int y = 0; y < 10; y++) {
			for (int x = 0; x < 10; x++) {
				int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
				delta_bias_outc += in[index]*25;
			}
		}
	}
	delta_bias[outc] = delta_bias_outc;
}

__kernel void  kernel_backward_c1(
	__global float *in, //delta_neuron_S2
	__global float *neuron_C1, //neuron_C1(in)
	__global float *weight_S2, //weight_S2(in) 
	__global float *delta_weight, // delta_weight_S2
	__global float *delta_bias,	 // delta_bias_S2
	__global float *out //delta_neuron_C1
){
	//[6,14,14]
	//[1,14,14]
	int outc = get_global_id(0);
	int y = get_global_id(1); 
	int x = get_global_id(2); 
	const float scale_factor = 0.25f;
	const int width_kernel_pooling_CNN = 2;
	const int height_kernel_pooling_CNN = 2;
	const int width_image_C1_CNN = 28;
	const int height_image_C1_CNN = 28;
	const int width_image_S2_CNN = 14;
	const int height_image_S2_CNN = 14;
	int block = 28*28*outc;
	int index = (outc*14*14) + y*14 + x;
	__local float w_tmp[14][14];
	__local float b_tmp[14][14];
	w_tmp[y][x] = 0;

	for (int m = 0; m < 2; m++) {
		for (int n = 0; n < 2; n++) {
			int addr2 = block + (y * 2 + m) * 28 + x * 2 + n;
			out[addr2] = in[index] * weight_S2[outc]
			* (1-neuron_C1[addr2]*neuron_C1[addr2]) * scale_factor;
			w_tmp[y][x]+=in[index] * neuron_C1[addr2] * scale_factor;
		}
	}
	b_tmp[y][x] = in[index] * 4;
	barrier(CLK_LOCAL_MEM_FENCE);
	if(x == 0 && y == 0){
		float tmpb=0,tmpw=0;
		for(int yy = 0;yy < 14;yy++)
			for(int xx = 0;xx < 14;xx++){
				tmpb += b_tmp[yy][xx];
				tmpw += w_tmp[yy][xx];
			}
		delta_weight[outc] = tmpw;
		delta_bias[outc] = tmpb;
	}
	// delta_weight[outc] += in[index] * neuron_C1[addr2] * scale_factor;
	// delta_bias[outc] += in[index];
}
__kernel void  kernel_backward_input(
	__global float *in, //delta_neuron_C1
	__global float *neuron_input, //data_single_image(in)
	__global float *weight_C1, //weight_C1(in) 
	// __global float *delta_weight, // delta_weight_C1
	// __global float *delta_bias,	 // delta_bias_C1
	__global float *out, //delta_neuron_input
	int index // index of data_single_image
){
	//[32, 32]
	int yy = get_global_id(0);
	int xx = get_global_id(1);
	__global float *data_single_image = neuron_input + index;
	// int width_image_input_CNN = 32;
	// int height_image_input_CNN = 32;
	// int width_image_C1_CNN = 28;
	// int height_image_C1_CNN = 28;
	// int width_kernel_conv_CNN = 5;
	// int height_kernel_conv_CNN = 5;
	// int num_map_C1_CNN = 6;
	int addr4 = yy*32+xx;
	float out_addr4 = 0;
	float data_single_image_addr4=data_single_image[addr4];

	for (int outc = 0; outc < 6; outc++) {
		int addr1 = 5*5*outc; //找到对应的卷积核
		for(int y = max(yy-4,0);y<=min(yy,27);y++){
			int wy = yy - y;
			for(int x = max(xx-4,0);x<=min(xx,27);x++){
				int wx = xx - x;
				int index = (outc*28*28) + y*28 + x; 
				int addr3 = addr1 + wy*5 + wx;  //卷积核索引 W_kj
				out_addr4 += in[index] * weight_C1[addr3] * (1-data_single_image_addr4*data_single_image_addr4);
			}
		}
	}
	out[addr4] = out_addr4;
}


__kernel void  kernel_backward_input_weight(
	__global float *in, //delta_neuron_C1
	__global float *neuron_input, //data_single_image(in)
	// __global float *weight_C1, //weight_C1(in) 
	__global float *delta_weight, // delta_weight_C1
	// __global float *delta_bias,	 // delta_bias_C1
	// __global float *out, //delta_neuron_input
	int index // index of data_single_image
){
	//[6,5,5]
	// int outc = get_global_id(0);
	// int wx = get_global_id(1);
	// int wy = get_global_id(2);
	// __global float *data_single_image = neuron_input + index;
	// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
	// float delta_weight_addr3 = 0;
	// for (int y = 0; y < 28; y++) {
	// 	for (int x = 0; x < 28; x++) {
	// 		int index = (outc*28*28) + y*28 + x;  //C1 当前神经元 j
	// 		int addr2 = y * 32 + x;  //input k
	// 		int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
	// 		delta_weight_addr3 += in[index] * data_single_image[addr4];
	// 	}
	// }
	// // printf("write:%d->%.6f\n", addr3,delta_weight_addr3);
	// delta_weight[addr3]=delta_weight_addr3;
	//[6,28,28]
	//[1,28,28]
	int outc = get_global_id(0);
	int y = get_global_id(1);
	int x = get_global_id(2);
	// printf("0:%d %d %d\n", outc, y, x);
	__global float *data_single_image = neuron_input + index;
	__local float w_tmp[28*28*15];
	int in_index = (outc*28*28) + y*28 + x;
	int addr2 = y * 32 + x;  //input k

	for (int wy = 0; wy < 3; wy++) {
		for (int wx = 0; wx < 5; wx++) {
			// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
			int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
			w_tmp[y*28*15+x*15+wy*5 + wx] = in[in_index] * data_single_image[addr4];
			// printf("write:%d,%d->%.6f\n", outc,y*28*15+x*15+wy*5 + wx,w_tmp[y*28*15+x*15+wy*5 + wx]);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(x == y && x < 15){
		private float tmp = 0;
		for(int i=0;i<28;i++){
			for(int j=0;j<28;j++){
				tmp += w_tmp[i*28*15+j*15+x];
			}
		}
		// printf("1:%d %d %d\n", outc, x, y);
		// printf("write:%d->%.6f\n", outc*25+x,tmp);
		delta_weight[outc*25+x] = tmp;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int wy = 3; wy < 5; wy++) {
		for (int wx = 0; wx < 5; wx++) {
			// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
			int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
			w_tmp[y*28*10+x*10+(wy-3)*5 + wx] = in[in_index] * data_single_image[addr4];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(x == y && x < 10){
		float tmp = 0;
		for(int i=0;i<28;i++){
			for(int j=0;j<28;j++){
				tmp += w_tmp[i*28*10+j*10+x];
			}
		}
		// printf("2:%d %d %d\n", outc, x, y);
		// printf("write:%d->%.6f\n", outc*25+x+15,tmp);
		delta_weight[outc*25+x+15] = tmp;
	}
}

__kernel void  kernel_backward_input_bias(
	__global float *in, //delta_neuron_C1
	// __global float *neuron_input, //data_single_image(in)
	// __global float *weight_C1, //weight_C1(in) 
	// __global float *delta_weight, // delta_weight_C1
	__global float *delta_bias	 // delta_bias_C1
	// __global float *out, //delta_neuron_input
	// int index // index of data_single_image
){
	//[6]
	int outc = get_global_id(0);
	// __global float *data_single_image = neuron_input + index;
	float delta_bias_outc = 0;
	for (int y = 0; y < 28; y++) {
		for (int x = 0; x < 28; x++) {
			int index = (outc*28*28) + y*28 + x;  //C1 当前神经元 j
			delta_bias_outc += in[index]*25;
		}
	}
	delta_bias[outc] = delta_bias_outc;
}

__kernel void kernel_update_weights(
	__global float * delta,
	__global float * e_weight,
	__global float * weight
){
	int i = get_global_id(0);
	float delta_tmp = delta[i];
	float e_weight_tmp = e_weight[i];
	e_weight_tmp += delta_tmp * delta_tmp;
	weight[i] -= 0.01 * delta_tmp / (sqrt(e_weight_tmp) + 1e-8);
	e_weight[i] = e_weight_tmp;
}
