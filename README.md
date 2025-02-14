A Perl based implementation of Principal Component Analysis and K-Means clustering.  Uses Inline::C to interface to both CUDA and ROCM SDKs.

Also included is a Perl implementation of a Scatter plot, using the Chart framework as a base.

An example script using the library on the Iris dataset is included.

To build the GPU libraries that this code uses, run install_gpu_modules.sh.  Depends on CUDA and/or ROCM SDK installed.  Tested on Debian 12.
