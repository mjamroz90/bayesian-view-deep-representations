In order to run DP-GMM model estimation, one must build a small native library (implementing Cholesky updates for cov matrices). 
To compile that lib, first enable ```tf_mkl``` environment:
```
conda activate tf_mkl
```
then go to:
```
src/models/clustering/lib/tf_cholesky_date
```
and follow instructions in README.md file therein.
