// MIT License
//
// Copyright (c) Mvine Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>

// HIP_CHECK copied from Common/example_utils.hpp
constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

constexpr unsigned int block_size = 16;
constexpr unsigned int BLOCK_SIZE = 16;


__global__ void gpu_transpose_2D_array(float *in, float *transposed, size_t rows, int cols);
   
__global__ void gpu_transpose_2D_array(float *in, float *transposed, size_t rows, size_t cols)
{  
        
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < cols && yIndex < rows)
   {  
       unsigned int index_in  = xIndex + cols * yIndex;
       unsigned int index_out = yIndex + rows * xIndex;
       transposed[index_out] = in[index_in]; 
   }
}     

__global__
void gpu_linear(float *a, float *b,  float *c, float *r, int m, int n, int k);
      
__global__ void gpu_linear(float *a, float *b,  float *c, float *r, int m, int n, int k)
{     
// linear = weights x activation + bias, so r = a x b + c (using the short param names)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum + c[row];
    }
}   

__global__
void gpu_matmul(float *a, float *b,  float *r, int m, int n, int k);

__global__ void gpu_matmul(float *a, float *b,  float *r, int m, int n, int k)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum ;
    }
}

__global__ 
void gpu_matrix_sigmoid(float *a, float *b, int m, int n );

__global__ void gpu_matrix_sigmoid(float *a,float *b, int m, int n )
{   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        b[row * n + col] = 1 / ( 1 + exp( -1 * a[row * n + col] ) );
    }
}   
    

__global__ void gpu_mse_cost(float *a, float *t, float *o, int m, int n );

__global__ void gpu_mse_cost(float *a, float *t, float *o, int m, int n ) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = powf(a[row * n + col] - t[row * n + col],2)/2;
    }
}

__global__ void gpu_cle_cost(float *a, float *t, float *o, int m, int n );

__global__ void gpu_cle_cost(float *a, float *t, float *o, int m, int n ) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        float arg1 = a[row * n + col];
        if (arg1 > 0) {
           arg1 = log(a[row * n + col]);
        } else {
           arg1 = 0;
        }
        float arg2 = 1 - a[row * n + col];
        if (arg2 > 0) {
           arg2 = log(1 - a[row * n + col]);
        } else {
           arg2 = 0;
        }
        o[row * n + col] = -t[row * n + col]*arg1-(1-t[row * n + col])*arg2;
    }
}

__global__ void gpu_mse_cost_derivative(float *a, float *t, float *o, int m, int n );
    
__global__ void gpu_mse_cost_derivative(float *a, float *t, float *o, int m, int n ) {
    
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = (a[row * n + col] - t[row * n + col]) * ( a[row * n + col] * ( 1 - a[row * n + col] ) );
    }
}   

__global__ void gpu_cle_cost_derivative(float *a, float *t, float *o, int m, int n );

__global__ void gpu_cle_cost_derivative(float *a, float *t, float *o, int m, int n ) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = (a[row * n + col] - t[row * n + col]);
    }
}

__global__ void gpu_add_two_same_size(float *a,float *b, size_t m, size_t n);
    
__global__ void gpu_add_two_same_size(float *a,float *b, size_t m, size_t n) {
    
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        a[row * n + col] += b[row * n + col];
    }
}   

__global__
void gpu_weight_derivative(float *a, float *b, float *r, int m, int n, int k);
    
__global__ void gpu_weight_derivative(float *a, float *b, float *r, int m, int n, int k)
{   
// weight_prime = delta x activation + current weight_prime, so r = a x b + r (using the short param names)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        { 
            sum += a[row * n + i] * b[i * k + col];
        } 
        r[row * k + col] = sum + r[row * k + col];
    }
} 

__global__ void gpu_sigmoid_prime(float *a, float *o, int m, int n );
    
__global__ void gpu_sigmoid_prime(float *a, float *o, int m, int n ) {
    
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] =  a[row * n + col] * ( 1 - a[row * n + col] ) ;
    }
}   
    
__global__
void gpu_derivative(float *a, float *b,  float *c, float *r, int m, int n, int k);
      
__global__ void gpu_derivative(float *a, float *b,  float *c, float *r, int m, int n, int k)
{        
// linear = weights x activation + bias, so r = a x b + c (using the short param names)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum * c[row * k + col];
    }
}   

__global__ void gpu_update_weights(float modifier, float decay, float *a,float *b, size_t m, size_t n);
    
__global__ void gpu_update_weights(float modifier, float decay, float *a,float *b, size_t m, size_t n) {
    
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        a[row * n + col] = decay * a[row * n + col ] - modifier * b[row * n + col];
    }
}   

__global__ void gpu_update_biases(float modifier, float *a,float *b, size_t m, size_t n);

__global__ void gpu_update_biases(float modifier, float *a,float *b, size_t m, size_t n) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < 1 && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += b[row * n + i];
        }
        a[row] -= modifier * sum;
    }
}

__global__ void gpu_calc_means(float *data, float *means, size_t height, size_t width);
    
__global__ void gpu_calc_means(float *data, float *means, size_t height, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < width  && row < 1) // add up each column, so only 1 row needed
    {
           for (size_t j = 0; j < height; j++) {
              means[col] += data[ j * width + col ] / height;
           } 
    }
}   

__global__ void gpu_calc_stddev(float *data, float *means, float *stddev, size_t height, size_t width);
    
__global__ void gpu_calc_stddev(float *data, float *means, float *stddev, size_t height, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < width  && row < 1) // add up each column, so only 1 row needed
    {
           for (size_t j = 0; j < height; j++) {
              stddev[col] += powf(data[ j * width + col ] - means[col], 2);
           } 
           stddev[col]  = sqrt( stddev[ col ] / (height - 1) );
    }
}   

__global__ void gpu_assign_z_scores(float *data, float *means, float *stddev, float *z, size_t height, size_t width);
    
__global__ void gpu_assign_z_scores(float *data, float *means, float *stddev, float *z, size_t height, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < width  && row < height)
    { 
           if (stddev[col] == 0) {
              z[row * width + col ] = 0;
           } else {
              z[row * width + col] = (data[row * width + col] - means[col]) / stddev[col];
              //z[row * width + col] = (data[row * width + col] - means[col]);// / stddev[col];
           }
    } 
}   
   
__global__ void gpu_calc_covariance(float *z, float *cov, size_t height, size_t width);
      
__global__ void gpu_calc_covariance(float *z, float *cov, size_t height, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < width  && row < width) 
    {    
           float sum = 0;
           if (col == row) {
              for (size_t i = 0; i < height; i++) {
                 sum += powf(z[i * width + col], 2) / height;
              } 
           } else {
              for (size_t i = 0; i < height; i++) {
                 sum += (z[i * width + col] * z[i * width + row]) / height;
              }
           }
           cov[ row * width + col] = sum ;
    }
}  

__global__ void gpu_qr_column_mult(float *orig, float *r, float *dotp, size_t height, size_t width, int this_column);

__global__ void gpu_qr_column_mult(float *orig, float *r, float *dotp, size_t height, size_t width, int this_column) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;  
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < this_column && row < 1)
    {
           float dotprod = 0;
           for (size_t j = 0; j < height; j++) {
               dotprod += orig[ j * width + this_column ] * r[j * width + col];
           } 
           dotp[ col ] = dotprod;
    }
}   

__global__ void gpu_qr_column(float *orig, float *r, float *dotp, size_t height, size_t width, int this_column);
    
__global__ void gpu_qr_column(float *orig, float *r, float *dotp, size_t height, size_t width, int this_column) {

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col == this_column && row < height)
    {
        float ax = orig[ row * width + this_column]; 
        r[ row * width + this_column ] = ax;
        for(size_t i = 0; i < this_column; i++)
        {
           float dotprod = dotp[ i ]; 
           //for (size_t j = 0; j < height; j++) {
            //   dotprod += r[ j * width + this_column ] * r[j * width + i];
           //}
           r[ row * width + this_column ] -= r[row * width + i] * dotprod;
        }
    }   
}

__global__ void gpu_qr_l2_norm(float *r, float *l2norm, size_t height, size_t width, int this_column);
    
__global__ void gpu_qr_l2_norm(float *r, float *l2norm, size_t height, size_t width, int this_column) {
// yes, this is single threading the GPU, but it does mean that we don't have to copy the whole matrix
// back to the host to work our the l2 norm of the column
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col == this_column && row < 1)
    {
        float l2norm = 0;
        for (int i = 0; i < height; i++) {
           l2norm += powf(r[i * width + this_column] , 2);
        }
        l2norm = sqrt(l2norm);
        for (int i = 0; i < height; i++) {
           r[i * width + this_column] /= l2norm; 
        }  
    }
}   

__global__ void gpu_qr_clamp_r_to_0(float *r, size_t height, size_t width);
    
__global__ void gpu_qr_clamp_r_to_0(float *r, size_t height, size_t width) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < width  && row < height)
    {
           if (col < row) {
              r[row * width + col] = 0;
           }
    }
}   

__global__ void gpu_eigenvector_signs(float *eigenvectors, float *sums, size_t rows, size_t cols);

__global__ void gpu_eigenvector_signs(float *eigenvectors, float *sums, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < cols  && row < 1) // one calc per vector
    {
        sums[col] = 0;
        for (int i = 0; i < rows; i++) {
              sums[col] += eigenvectors[i * cols + col];
        }
        if (sums[col] < 1) {
           sums[col] = 0;
           for (int i = 0; i < rows; i++) {
              eigenvectors[i * cols + col] *= -1;
              sums[col] += eigenvectors[i * cols + col];
           }
        }
    }
}   

__global__ void gpu_reorder_eigenvectors(float *unsorted, float *sorted, int *indicies, size_t rows, size_t cols);

__global__ void gpu_reorder_eigenvectors(float *unsorted, float *sorted, int *indicies, size_t rows, size_t cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < cols  && row < rows)
    {
       sorted[row * cols + col] = unsorted[row * cols + indicies[col]];
    }
}

int run_gpu_linear( float *activation, float *device_Weights,  float *device_Bias, float *device_Output, int m, int n, int k) ;
int run_gpu_linear( float *activation, float *device_Weights,  float *device_Bias, float *device_Output, int m, int n, int k) {
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_linear<<<dimGridT, dimBlockT>>>(device_Weights, activation, device_Bias, device_Output,m, n, k);
    return 1;
}  

int run_gpu_matmul( float *a, float *b,  float *r, int input_size, int middle_size, int output_size) ;
int run_gpu_matmul( float *a, float *b,  float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matmul<<<dimGridT, dimBlockT>>>(a, b, r, input_size,middle_size, output_size);
    return 1;
}

int run_gpu_derivative( float *a, float *b,  float *sp, float *r, int input_size, int middle_size, int output_size) ;
int run_gpu_derivative( float *a, float *b,  float *sp, float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_derivative<<<dimGridT, dimBlockT>>>(a, b, sp, r,input_size, middle_size, output_size);
    return 1;
}   

int run_gpu_weight_derivative( float *a, float *b, float *r, int input_size, int middle_size, int output_size);
int run_gpu_weight_derivative( float *a, float *b, float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_weight_derivative<<<dimGridT, dimBlockT>>>(a, b, r,input_size, middle_size, output_size);
    return 1;
}   

int gpu_sigmoid( float *device_Output, float *device_Activated_Output, int output_size , int cols);
int gpu_sigmoid( float *device_Output, float *device_Activated_Output, int output_size , int cols) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    
    gpu_matrix_sigmoid<<<dimGrid, dimBlock>>>(device_Output, device_Activated_Output, output_size, cols);
    return 1;
}       

int run_gpu_sigmoid_prime( float *device_Activated_Output, float *device_Activated_Output_Derivative, int output_size, int cols );
int run_gpu_sigmoid_prime( float *device_Activated_Output, float *device_Activated_Output_Derivative, int output_size, int cols ) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_sigmoid_prime<<<dimGrid, dimBlock>>>(device_Activated_Output, device_Activated_Output_Derivative, output_size, cols);
    return 1;
}   

int gpu_add_same_size( float *lhs, float *rhs, int arraysize ); 
int gpu_add_same_size( float *lhs, float *rhs, int arraysize ) {
    unsigned int grid_rows = (arraysize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_add_two_same_size<<<dimGrid, dimBlock>>>(lhs, rhs, arraysize, 1);
    return 1;
}   

void run_gpu_transpose_2D_array( float *in, float *transpose, size_t rows, size_t cols) ;
void run_gpu_transpose_2D_array( float *in, float *transpose, size_t rows, size_t cols) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_transpose_2D_array<<<dimGrid, dimBlock>>>(in, transpose, rows, cols);
}   

void run_gpu_update_weights( float modifier, float decay, float *device_Weights, float *device_Weights_Derivative, int output_size, int input_size) ;
void run_gpu_update_weights( float modifier, float decay, float *device_Weights, float *device_Weights_Derivative, int output_size, int input_size) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    gpu_update_weights<<<dimGrid, dimBlock>>>(modifier, decay, device_Weights, device_Weights_Derivative, output_size, input_size);
}   

void run_gpu_update_biases( float modifier, float *device_Bias, float *device_Bias_Derivative, int output_size, int batch_size);

void run_gpu_update_biases( float modifier, float *device_Bias, float *device_Bias_Derivative, int output_size, int batch_size){
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    gpu_update_biases<<<dimGrid, dimBlock>>>(modifier, device_Bias, device_Bias_Derivative, output_size, batch_size);
}   
    

void gpu_calculate_cost_and_derivative(float *activated_output, float *y, float *cost_derivative, size_t rows, size_t cols, int loss_function) ;
void gpu_calculate_cost_and_derivative(float *activated_output, float *y, float *cost_derivative, size_t rows, size_t cols, int loss_function) { 
    // (output - target) * output_derivative : output_derivative can be derived from output, so do it all within the GPU
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if (loss_function == 2) {
       gpu_cle_cost_derivative<<<dimGrid, dimBlock>>>( activated_output, y, cost_derivative, rows, cols );
    } else {
       gpu_mse_cost_derivative<<<dimGrid, dimBlock>>>( activated_output, y, cost_derivative, rows , cols );
    }
}

void gpu_calculate_cost(float *device_Activated_Output, float *device_y, float *device_Cost, size_t rows, size_t cols, int loss_function);
void gpu_calculate_cost(float *device_Activated_Output, float *device_y, float *device_Cost, size_t rows, size_t cols, int loss_function){
    // (output - target) * output_derivative : output_derivative can be derived from output, so do it all within the GPU
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    if (loss_function == 2) {
       gpu_cle_cost<<<dimGrid, dimBlock>>>( device_Activated_Output, device_y, device_Cost, rows, cols );
    } else {
       gpu_mse_cost<<<dimGrid, dimBlock>>>( device_Activated_Output, device_y, device_Cost, rows, cols );
    } 
}      

void run_gpu_calc_means( float *data, float *means, size_t rows, size_t cols ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    gpu_calc_means<<<dimGrid, dimBlock>>>(data, means, rows, cols);
}   

void run_gpu_calc_stddev( float *data, float *means, float *stddev, size_t rows, size_t cols ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_calc_stddev<<<dimGrid, dimBlock>>>(data, means, stddev, rows, cols);
}   

void run_gpu_assign_z_scores( float *data, float *means, float *stddev, float *z, size_t rows, size_t cols ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_assign_z_scores<<<dimGrid, dimBlock>>>(data, means, stddev, z, rows, cols);
}   

void run_gpu_calc_covariance( float *z, float *cov, size_t rows, size_t cols ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_calc_covariance<<<dimGrid, dimBlock>>>(z, cov, rows, cols);
}          

void run_gpu_qr_column_mult( float *orig, float *r, float *dotp, size_t rows, size_t cols, int colno ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_qr_column_mult<<<dimGrid, dimBlock>>>(orig, r, dotp, rows, cols, colno);
}  

void run_gpu_qr_column( float *orig, float *r, float *dotp, size_t rows, size_t cols, int colno ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows); 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_qr_column<<<dimGrid, dimBlock>>>(orig, r, dotp, rows, cols, colno);
}

void run_gpu_qr_l2_norm( float *r, float *l2norm, size_t rows, size_t cols, int colno ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_qr_l2_norm<<<dimGrid, dimBlock>>>(r, l2norm, rows, cols, colno);
}   

void run_gpu_qr_clamp_r_to_0( float *r, size_t rows, size_t cols ) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_qr_clamp_r_to_0<<<dimGrid, dimBlock>>>(r, rows, cols);
}          

void run_gpu_eigenvector_signs(float *eigenvectors, float *sums, size_t rows, size_t cols); 
void run_gpu_eigenvector_signs(float *eigenvectors, float *sums, size_t rows, size_t cols) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_eigenvector_signs<<<dimGrid, dimBlock>>>(eigenvectors, sums, rows, cols);
}


void run_gpu_reorder_eigenvectors(float *unsorted, float *sorted, int *indicies, size_t rows, size_t cols);
void run_gpu_reorder_eigenvectors(float *unsorted, float *sorted, int *indicies, size_t rows, size_t cols) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_reorder_eigenvectors<<<dimGrid, dimBlock>>>(unsorted, sorted, indicies, rows, cols);
}
    
void gpu_memcpy_to_device( float *host_data, float *device_data, size_t size_data);
void gpu_memcpy_to_device( float *host_data, float *device_data, size_t size_data) {
       HIP_CHECK(hipMemcpy(device_data, host_data, size_data, hipMemcpyHostToDevice));
}

void gpu_memcpy_to_device_int( int *host_data, int *device_data, size_t size_data);
void gpu_memcpy_to_device_int( int *host_data, int *device_data, size_t size_data) {
       HIP_CHECK(hipMemcpy(device_data, host_data, size_data, hipMemcpyHostToDevice));
}   

void gpu_memcpy_from_device( float *host_data, float *device_data, size_t size_data);
void gpu_memcpy_from_device( float *host_data, float *device_data, size_t size_data) {
       HIP_CHECK(hipMemcpy(host_data, device_data, size_data, hipMemcpyDeviceToHost));
}

void gpu_memcpy_intra_device( float *from_data, float *to_data, size_t size_data);
void gpu_memcpy_intra_device( float *from_data, float *to_data, size_t size_data) {
       HIP_CHECK(hipMemcpy(to_data, from_data, size_data, hipMemcpyDeviceToDevice));
}

float * gpu_device_malloc( size_t size_data);
float * gpu_device_malloc( size_t size_data) {
       float * device_data;
       HIP_CHECK(hipMalloc((void**)&device_data, size_data));
       return device_data;
}

float * gpu_host_malloc( size_t size_data);
float * gpu_host_malloc( size_t size_data) {
   float *host_data;
   //host_data = (float *)malloc( size_data + 1);
   HIP_CHECK(hipHostMalloc((void**)&host_data, size_data));
   return host_data;
}

int * gpu_device_malloc_int( size_t size_data);
int * gpu_device_malloc_int( size_t size_data) {
       int * device_data;
       HIP_CHECK(hipMalloc((void**)&device_data, size_data));
       return device_data;
}

void gpu_free_device(void *device_data) ;
void gpu_free_device(void *device_data) {
   HIP_CHECK(hipFree((void *)device_data));
}
void gpu_free_host(void *host_data) ;
void gpu_free_host(void *host_data) {
   HIP_CHECK(hipHostFree((void *)host_data));
}

