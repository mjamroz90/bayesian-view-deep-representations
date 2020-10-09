cmake_minimum_required(VERSION 3.5)

option(WITH_GPU_SUPPORT "With GPU support" ON)

#get the include directory for tensorflow
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

message("tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link dir: ${Tensorflow_LIB_DIRS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories("/usr/local/")
link_directories(${Tensorflow_LIB_DIRS})

if(WITH_GPU_SUPPORT)
    find_package(CUDA)
endif()

#set flags based on tutorial
if (WITH_GPU_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -DNDEBUG")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr" )
    SET(CUDA_PROPAGATE_HOST_FLAGS ON)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=0 -DNDEBUG")
endif()

set(CMAKE_BUILD_TYPE Debug)

include_directories(include)

#create library
if (WITH_GPU_SUPPORT)
    cuda_add_library(
        cholesky_date SHARED
        src/cholesky_date.cu
        src/cholesky_date.cc
    )
else()
    add_library(
        cholesky_date SHARED
        src/cholesky_date.cc
    )
endif()

target_link_libraries(cholesky_date "tensorflow_framework")
