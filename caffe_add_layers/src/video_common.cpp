
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth: Zhaofan Qiu
** mail: zhaofanqiu@gmail.com
** date: 2015/12/13
** desc: Caffe-video common
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/video_common.hpp"

namespace caffe {
	using namespace std;
	vector<int> video_shape(int num, int channels, int length, int height, int width)
	{
		vector<int> shape(5, 0);
		shape[0] = num;
		shape[1] = channels;
		shape[2] = length;
		shape[3] = height;
		shape[4] = width;
		return shape;
	}
}  // namespace caffe

