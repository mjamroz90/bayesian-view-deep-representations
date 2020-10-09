#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include <iostream>

#if GOOGLE_CUDA == 1
#include <cuda.h>
#endif

#include "cholesky_date.h"

using namespace tensorflow;
using namespace std;
using namespace shape_inference;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

//For now only accept 3 and 2 for rank
//i.e. batch of matrices and batch of vectors
Status ShapeFn(InferenceContext* c)
{
	//check input shape has 2 dimensions (d, d)
	ShapeHandle r_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &r_shape));

	//check indices has 1 dimension (d)
	ShapeHandle x_shape;
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x_shape));

	int r_rank = c->Rank(r_shape);
	int x_rank = c->Rank(x_shape);
	//R must be square
	if (c->Value(c->Dim(r_shape,r_rank - 1)) != c->Value(c->Dim(r_shape, r_rank - 2)))
			return errors::InvalidArgument("R must be square");


    DimensionHandle r_dim = c->Dim(r_shape,0);
    DimensionHandle x_dim = c->Dim(x_shape,0);

    if (c->Value(r_dim) != c->Value(x_dim))
        return errors::InvalidArgument("R and x must have same dims");

	//set output size
	c->set_output(0, c->input(0));

	return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("CholUpdate")
		.Input("l: T")
		.Input("x: T")
		.Output("c: T")
        .Attr(GetConvnetDataFormatAttrString())
		.Attr("T: {float32, float64}")
		.Attr("use_locking: bool = false")
		.SetShapeFn(ShapeFn);

REGISTER_OP("CholDowndate")
		.Input("l: T")
		.Input("x: T")
		.Output("c: T")
        .Attr(GetConvnetDataFormatAttrString())
		.Attr("T: {float32, float64}")
		.Attr("use_locking: bool = false")
		.SetShapeFn(ShapeFn);


template <typename T>
struct launchCholUpdateKernel<CPUDevice, T> {
	void operator()(const CPUDevice& d, typename TTypes<T>::Flat L_out,
			typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x,
            typename TTypes<T>::ConstFlat x_in, int dim) {

		x.setZero();
		L_out.setZero();

		x += x_in;
		L_out += L;

		for(int k = 0; k < dim; k++)
		{
			T Lkk = L_out(LIND(k,k));
			T xk = x(k);

			T r = sqrt(Lkk*Lkk + xk*xk);
			T c = r/Lkk;
			T s = xk/Lkk;
			L_out(LIND(k,k)) = r;
			for(int i = k+1; i < dim; i++)
			{
				L_out(LIND(i,k)) = (L_out(LIND(i,k)) + s*x(i)) / c;
				x(i) = c*x(i) - s*L_out(LIND(i,k));
			}
		}

	}
};

template <typename T>
struct launchCholDowndateKernel<CPUDevice, T> {
	void operator()(const CPUDevice& d, typename TTypes<T>::Flat L_out,
			typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x,
            typename TTypes<T>::ConstFlat x_in, int dim) {

		x.setZero();
		L_out.setZero();

		x += x_in;
		L_out += L;

		for(int k = 0; k < dim; k++)
		{
			T Lkk = L_out(LIND(k,k));
			T xk = x(k);

			T r = sqrt(Lkk*Lkk - xk*xk);
			T c = r/Lkk;
			T s = xk/Lkk;
			L_out(LIND(k,k)) = r;
			for(int i = k+1; i < dim; i++)
			{
				L_out(LIND(i,k)) = (L_out(LIND(i,k)) - s*x(i)) / c;
				x(i) = c*x(i) - s*L_out(LIND(i,k));
			}
		}

	}
};

template <typename Device, typename T>
class CholUpdateOp : public OpKernel {
public:

	explicit CholUpdateOp(OpKernelConstruction* context)
		: OpKernel(context)
	{
		string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));

		//only nhwc supported
        OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("CholUpdate only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));

		OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));

	}

	void Compute(OpKernelContext* context) override {
		// We always return the input ref.
    	//context->forward_ref_input_to_ref_output(0, 0);

		if (use_exclusive_lock_) {
			mutex_lock l(*context->input_ref_mutex(0));
			DoUpdate(context);
		} else {
			DoUpdate(context);
		}
	}
private:
	void DoUpdate(OpKernelContext* context) {
		// Grab the input tensor
		const Tensor& l_tensor = context->input(0);
		const Tensor& x_tensor = context->input(1);

		Tensor x_workspace;
		Tensor* l_out = NULL;
    	OP_REQUIRES_OK(context, context->allocate_temp(
							DataTypeToEnum<T>::v(),
							x_tensor.shape(), &x_workspace));

		OP_REQUIRES_OK(context, context->allocate_output(0, l_tensor.shape(), &l_out));

		int dim = GetTensorDim(l_tensor.shape(), data_format_, 'H');

		//flatten tensors
		auto x_work_flat = x_workspace.flat<T>();
		auto l_flat = l_tensor.flat<T>();
		auto l_out_flat = l_out->flat<T>();
		auto x_flat = x_tensor.flat<T>();

		// Call the cuda kernel launcher
		launchCholUpdateKernel<Device, T>()(
			context->eigen_device<Device>(),
			l_out_flat,
			l_flat,
			x_work_flat,
			x_flat,
			dim);
	}
    TensorFormat data_format_;
	bool use_exclusive_lock_;
};

/**Repeating everything from class above - nasty as shit, but refactor will be later*/
template <typename Device, typename T>
class CholDowndateOp : public OpKernel {
public:

	explicit CholDowndateOp(OpKernelConstruction* context)
		: OpKernel(context)
	{
		string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));

		//only nhwc supported
        OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("CholDowndate only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));

		OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));

	}

	void Compute(OpKernelContext* context) override {
		// We always return the input ref.
    	//context->forward_ref_input_to_ref_output(0, 0);

		if (use_exclusive_lock_) {
			mutex_lock l(*context->input_ref_mutex(0));
			DoDowndate(context);
		} else {
			DoDowndate(context);
		}
	}
private:
	void DoDowndate(OpKernelContext* context) {
		// Grab the input tensor
		const Tensor& l_tensor = context->input(0);
		const Tensor& x_tensor = context->input(1);

		Tensor x_workspace;
		Tensor* l_out = NULL;
    	OP_REQUIRES_OK(context, context->allocate_temp(
							DataTypeToEnum<T>::v(),
							x_tensor.shape(), &x_workspace));
		OP_REQUIRES_OK(context, context->allocate_output(0, l_tensor.shape(), &l_out));

		int dim = GetTensorDim(l_tensor.shape(), data_format_, 'H');

		//flatten tensors
		auto x_work_flat = x_workspace.flat<T>();
		auto l_flat = l_tensor.flat<T>();
		auto l_out_flat = l_out->flat<T>();
		auto x_flat = x_tensor.flat<T>();

		// Call the cuda kernel launcher
		launchCholDowndateKernel<Device, T>()(
			context->eigen_device<Device>(),
			l_out_flat,
			l_flat,
			x_work_flat,
			x_flat,
			dim);
	}
    TensorFormat data_format_;
	bool use_exclusive_lock_;
};

//register kernel with types needed
#if GOOGLE_CUDA == 1
#define REGISTER_UP_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		CholUpdateOp<GPUDevice, type>) \

REGISTER_UP_GPU(float);
REGISTER_UP_GPU(double);


#undef REGISTER_UP_GPU
#endif

#if GOOGLE_CUDA == 1
#define REGISTER_DOWN_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholDowndate") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		CholDowndateOp<GPUDevice, type>) \

REGISTER_DOWN_GPU(float);
REGISTER_DOWN_GPU(double);


#undef REGISTER_DOWN_GPU
#endif

#define REGISTER_UP_CPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholUpdate") \
		.Device(DEVICE_CPU) \
		.TypeConstraint<type>("T"), \
		CholUpdateOp<CPUDevice, type>) \

REGISTER_UP_CPU(float);
REGISTER_UP_CPU(double);

#undef REGISTER_UP_CPU

#define REGISTER_DOWN_CPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("CholDowndate") \
		.Device(DEVICE_CPU) \
		.TypeConstraint<type>("T"), \
		CholDowndateOp<CPUDevice, type>) \

REGISTER_DOWN_CPU(float);
REGISTER_DOWN_CPU(double);

#undef REGISTER_DOWN_CPU