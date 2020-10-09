#pragma once
#if GOOGLE_CUDA == 1
#define EIGEN_USE_GPU
#else
#define EIGEN_USE_CPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorForwardDeclarations.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#define LIND(x,y) ((x))*dim + (y)

using namespace tensorflow;

template <typename Device, typename T>
struct launchCholUpdateKernel {
  void operator()(const Device& d, typename TTypes<T>::Flat L_out,
            typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x_workspace,
            typename TTypes<T>::ConstFlat x, int dim);

};

template <typename Device, typename T>
struct launchCholDowndateKernel {
  void operator()(const Device& d, typename TTypes<T>::Flat L_out,
            typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x_workspace,
            typename TTypes<T>::ConstFlat x, int dim);
};