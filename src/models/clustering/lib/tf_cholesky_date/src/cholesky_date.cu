#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <cmath>
#include "tensorflow/core/util/cuda_device_functions.h"

#include "cholesky_date.h"

using GPUDevice = Eigen::GpuDevice;


template <typename T>
__global__ void DoLoopKernel(T* L, T* x, int dim, int k, int sign, T s, T c){

    int i_indx = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i_indx < dim && i_indx > k){

        L[LIND(i_indx, k)] = (L[LIND(i_indx, k)] + sign*s*x[i_indx])/c;
        x[i_indx] = c*x[i_indx] - s*L[LIND(i_indx, k)];
    }

}


template <typename T> void CholDate(T* L, T* x, int dim, int sign, const GPUDevice& d){

    for (int k=0; k<dim; k++){
		T Lkk, xk;

		cudaMemcpy(&Lkk, &L[LIND(k, k)], sizeof(T), cudaMemcpyDeviceToHost);

		cudaMemcpy(&xk, &x[k], sizeof(T), cudaMemcpyDeviceToHost);

		T r = sqrt(Lkk*Lkk + sign*xk*xk);
		T c = r/Lkk;
		T s = xk/Lkk;

		cudaMemcpy(&L[LIND(k,k)], &r, sizeof(T), cudaMemcpyHostToDevice);

		const int kThreadsPerBlock = 1024;
		const int numBlocks = ceil((dim - k) / (float)kThreadsPerBlock);

		DoLoopKernel<T><<<numBlocks, kThreadsPerBlock, 0, d.stream()>>>(L, x, dim, k, sign, s, c);
		cudaDeviceSynchronize();

    }
}


template <typename T>
struct launchCholUpdateKernel<GPUDevice, T> {
	void operator()(const GPUDevice& d, typename TTypes<T>::Flat L_out,
			typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x,
			typename TTypes<T>::ConstFlat x_in, int dim) {

		To32Bit(x).device(d) = To32Bit(x_in);
		To32Bit(L_out).device(d) = To32Bit(L);

		CholDate<T>(L_out.data(), x.data(), dim, 1, d);

		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	}
};

template <typename T>
struct launchCholDowndateKernel<GPUDevice, T> {
	void operator()(const GPUDevice& d, typename TTypes<T>::Flat L_out,
			typename TTypes<T>::ConstFlat L, typename TTypes<T>::Flat x,
			typename TTypes<T>::ConstFlat x_in, int dim) {

		To32Bit(x).device(d) = To32Bit(x_in);
		To32Bit(L_out).device(d) = To32Bit(L);

		CholDate<T>(L_out.data(), x.data(), dim, -1, d);

		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
			printf("kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
	}
};


//forward declaration for all the types needed
typedef Eigen::GpuDevice GPUDevice;
#define ADD_KERNEL_UP_TYPE(type)							\
	template struct launchCholUpdateKernel<GPUDevice, type>;	\

ADD_KERNEL_UP_TYPE(float);
ADD_KERNEL_UP_TYPE(double);


#undef ADD_KERNEL_UP_TYPE


#define ADD_KERNEL_DOWN_TYPE(type)							\
	template struct launchCholDowndateKernel<GPUDevice, type>;	\

ADD_KERNEL_DOWN_TYPE(float);
ADD_KERNEL_DOWN_TYPE(double);


#undef ADD_KERNEL_DOWN_TYPE