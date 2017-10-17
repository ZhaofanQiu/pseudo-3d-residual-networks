
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： Pooling3D layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/pooling3d_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int length, const int height, const int width, const int pooled_length, const int pooled_height,
    const int pooled_width, const int kernel_l, const int kernel_h, const int kernel_w,
    const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
	const int pl = (index / pooled_width / pooled_height) % pooled_length;
    const int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    const int n = index / pooled_width / pooled_height / pooled_length / channels;
    int lstart = pl * stride_l - pad_l;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int lend = min(lstart + kernel_l, length);
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    lstart = max(lstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * length * height * width;
	for (int l = lstart; l < lend; ++l) {
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (bottom_slice[l * height * width + h * width + w] > maxval) {
					maxidx = l * height * width + h * width + w;
					maxval = bottom_slice[maxidx];
				}
			}
		}
	}
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int length, const int height, const int width, const int pooled_length, const int pooled_height,
    const int pooled_width, const int kernel_l, const int kernel_h, const int kernel_w,
    const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int pl = (index / pooled_width / pooled_height) % pooled_length;
    const int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    const int n = index / pooled_width / pooled_height / pooled_length / channels;
    int lstart = pl * stride_l - pad_l;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int lend = min(lstart + kernel_l, length + pad_l);
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (lend - lstart) * (hend - hstart) * (wend - wstart);
    lstart = max(lstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    lend = min(lend, length);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * length * height * width;
	for (int l = lstart; l < lend; ++l) {
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				aveval += bottom_slice[l * height * width + h * width + w];
			}
		}
	}
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int pl = (index / pooled_width / pooled_height) % pooled_length;
    const int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    const int n = index / pooled_width / pooled_height / pooled_length / channels;
    const int lstart = pl * stride_l;
    const int lend = min(lstart + kernel_l, length);
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * length * height * width;
    // First pass: get sum
	for (int l = lstart; l < lend; ++l) {
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				cumsum += bottom_slice[l * height * width + h * width + w];
			}
		}
	}
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int l = lstart; l < lend; ++l) {
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				cumsum += bottom_slice[l * height * width + h * width + w];
				if (cumsum >= thres) {
					rand_idx[index] = (((n * channels + c) * length + l) * height + h) * width + w;
					top_data[index] = bottom_slice[l * height * width + h * width + w];
					return;
				}
			}
		}
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
	const Dtype* const bottom_data,
	const int num, const int channels, const int length, const int height,
	const int width, const int pooled_length, const int pooled_height, const int pooled_width,
	const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
	const int stride_w, Dtype* const top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		const int pl = (index / pooled_width / pooled_height) % pooled_length;
		const int c = (index / pooled_width / pooled_height / pooled_length) % channels;
		const int n = index / pooled_width / pooled_height / pooled_length / channels;;
		const int lstart = pl * stride_l;
		const int lend = min(lstart + kernel_l, length);
		const int hstart = ph * stride_h;
		const int hend = min(hstart + kernel_h, height);
		const int wstart = pw * stride_w;
		const int wend = min(wstart + kernel_w, width);
		// We set cumsum to be 0 to avoid divide-by-zero problems
		Dtype cumsum = FLT_MIN;
		Dtype cumvalues = 0.;
		const Dtype* const bottom_slice =
			bottom_data + (n * channels + c) * length * height * width;;
		// First pass: get sum
		for (int l = lstart; l < lend; ++l) {
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					cumsum += bottom_slice[l * height * width + h * width + w];
					cumvalues += bottom_slice[l * height * width + h * width + w] * bottom_slice[l * height * width + h * width + w];
				}
			}
		}
		top_data[index] = cumvalues / cumsum;
	}
}


template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling3d_param().pool()) {
  case Pooling3DParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_,
        length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_l_, kernel_h_,
        kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case Pooling3DParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_,
        length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_l_, kernel_h_,
        kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_, top_data);
    break;
  case Pooling3DParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->shape(0), channels_,
          length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_l_, kernel_h_,
          kernel_w_, stride_l_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->shape(0), channels_,
          length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_l_, kernel_h_,
          kernel_w_, stride_l_, stride_h_, stride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int length, const int height, const int width,
    const int pooled_length, const int pooled_height, const int pooled_width,const int kernel_l, const int kernel_h,
    const int kernel_w, const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int l = (index / width / height) % length;
    const int c = (index / width / height / length) % channels;
    const int n = index / width / height / length / channels;
    const int plstart =
         (l + pad_l < kernel_l) ? 0 : (l + pad_l - kernel_l) / stride_l + 1;
    const int plend = min((l + pad_l) / stride_l + 1, pooled_length);
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_length * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
	  for (int pl = plstart; pl < plend; ++pl) {
		  for (int ph = phstart; ph < phend; ++ph) {
			  for (int pw = pwstart; pw < pwend; ++pw) {
				  if (mask_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw] == l * height * width + h * width + w) {
					  gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw];
				  }
			  }
		  }
	  }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
	  for (int pl = plstart; pl < plend; ++pl) {
		  for (int ph = phstart; ph < phend; ++ph) {
			  for (int pw = pwstart; pw < pwend; ++pw) {
				  if (top_mask_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw] == l * height * width + h * width + w) {
					  gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw];
				  }
			  }
		  }
	  }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
	const int num, const int channels, const int length, const int height, const int width,
	const int pooled_length, const int pooled_height, const int pooled_width, const int kernel_l, const int kernel_h,
	const int kernel_w, const int stride_l, const int stride_h, const int stride_w, const int pad_l, const int pad_h,
	const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	  const int w = index % width;
	  const int h = (index / width) % height;
	  const int l = (index / width / height) % length;
	  const int c = (index / width / height / length) % channels;
	  const int n = index / width / height / length / channels;
	  const int plstart =
		  (l + pad_l < kernel_l) ? 0 : (l + pad_l - kernel_l) / stride_l + 1;
	  const int plend = min((l + pad_l) / stride_l + 1, pooled_length);
	  const int phstart =
		  (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
	  const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
	  const int pwstart =
		  (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
	  const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_length * pooled_height * pooled_width;
	for (int pl = plstart; pl < plend; ++pl) {
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				// figure out the pooling size
				int lstart = pl * stride_l - pad_l;
				int hstart = ph * stride_h - pad_h;
				int wstart = pw * stride_w - pad_w;
				int lend = min(lstart + kernel_l, length + pad_l);
				int hend = min(hstart + kernel_h, height + pad_h);
				int wend = min(wstart + kernel_w, width + pad_w);
				int pool_size = (lend - lstart) * (hend - hstart) * (wend - wstart);
				gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw] / pool_size;
			}
		}
	}
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_l, const int kernel_h, const int kernel_w, const int stride_l, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
	  const int w = index % width;
	  const int h = (index / width) % height;
	  const int l = (index / width / height) % length;
	  const int c = (index / width / height / length) % channels;
	  const int n = index / width / height / length / channels;
    const int plstart = (l < kernel_l) ? 0 : (l - kernel_l) / stride_l + 1;
    const int plend = min(l / stride_l + 1, pooled_length);
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_length * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_length * pooled_height * pooled_width;
	for (int pl = plstart; pl < plend; ++pl) {
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw] *
					(index == static_cast<int>(rand_idx_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw]));
			}
		}
	}
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling3d_param().pool()) {
  case Pooling3DParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->shape(0), channels_,
        length_, height_, width_, pooled_length_, pooled_height_, pooled_width_,
        kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case Pooling3DParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->shape(0), channels_,
        length_, height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_l_, kernel_h_,
        kernel_w_, stride_l_, stride_h_, stride_w_, pad_l_, pad_h_, pad_w_, bottom_diff);
    break;
  case Pooling3DParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->shape(0), channels_, length_, height_, width_, pooled_length_, pooled_height_,
        pooled_width_, kernel_l_, kernel_h_, kernel_w_, stride_l_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(Pooling3DLayer);
}  // namespace caffe
