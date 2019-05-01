__constant int tbl[6][16] = {
	{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
	{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
	{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
	{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
	{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
	{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
};

__kernel void  kernel_forward_c1(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out
					  )
{
	int channel = get_global_id(0);
	int out_height = get_global_size(1);
	int out_width = get_global_size(2);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 32;
	int in_height = 32;
	int in_num = 1;
    int index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	int out_val = 0.0;
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
	     out_val += sum;
	}
	out_val += bias[channel];
	out_val = tanh((float)(out_val));
	out[index] = out_val;
}

__kernel void  kernel_forward_s2(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
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
                      __global float  *out
					  )
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
	int out_index = 0.0;
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
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
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

// __kernel void  kernel_forward_c5(__global float *in,
//                       __global float  *weight,
//                       __global float  *bias,
//                       __global float  *out,
//                       int channel,
//                       int out_width,
//                       int out_height,
//                       int kernel_width,
// 					  int kernel_height,
// 					  int in_num,
// 					  int in_width,
//                       int in_height
// 					  )
// {
// 	channel = get_global_id(0);
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
//     int  index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	out[index] = 0.0;
// 	for (inc=0; inc<in_num; inc++) {
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*in_width*in_height;
// 		const float* pw = weight + addr1;   //卷积核
// 		const float* pi = in + addr2;       //输入图像
// 		sum = 0.0;
// 		const float* ppw = pw;
// 		const float* ppi = pi + y * in_width + x;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * in_width + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// }
