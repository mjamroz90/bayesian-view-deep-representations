# tf_cholesky_date
The [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) is defined as the decompostition of a matrix `A` into the form `A = LL^T`, or equivalently `A = R^TR` where `L = R^T`. Where `L` is lower triangular with real positive diagonal entries (upper triangular for `R`).

This repo implements:
- The [Rank One Update](https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update) of the Cholesky decompostion. The update rule is for a matrix `A' = A + xx^T`, can be defined in terms of `L`.
- The [Rank One Downdate]()

## How to build

Run the following commands after cloning this repo:

```
cd /path/to/project/src/models/clustering/lib/tf_cholesky_date
mkdir build
cd build
cmake .. -DWITH_GPU_SUPPORT=OFF
make
```

It's possible that you'll receive an error related to linking against tensorflow, this is because of wrong formatting of the command prepared by cmake. 
Invoke the command manually from CLI (in build directory), ex.:
```
cat CMakeFiles/cholesky_date.dir/link.txt

/Library/Developer/CommandLineTools/usr/bin/c++  --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=0 -DNDEBUG -g -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -dynamiclib -Wl,-headerpad_max_install_names  -o libcholesky_date.dylib -install_name @rpath/libcholesky_date.dylib CMakeFiles/cholesky_date.dir/src/cholesky_date.cc.o  -L/Users/anonym/anaconda3/envs/tf_mkl/lib/python3.7/site-packages/tensorflow
 -Wl,-rpath,/Users/anonym/anaconda3/envs/tf_mkl/lib/python3.7/site-packages/tensorflow
 -ltensorflow_framework
```

Check if there appeared file `libcholesky_date.so` or `libcholesky_date.dylib` in build directory.
