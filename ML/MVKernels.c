#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "MVKernels.h"
#include "node_typedef.h"

int debug = 0;
int loss_function = 1;

int mini_batch_size;

#define HAVE_PERL_VERSION(R, V, S) \
    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

float *host_Cost_Derivative;
float *device_Cost_Derivative;
float *host_Cost;
float *device_Cost;
int dirty_dirty_weights = 0;

// This section is boilerplace code to move data from Perl -> C and back again

#define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)

static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
{
  sv_setiv(sv, (IV)rv);
#if !HAVE_PERL_VERSION(5, 24, 0)
  SvIOK_off(sv);
#endif
  SvROK_on(sv);
}

int is_array_ref(
        SV *array,
        size_t *array_sz
);
int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
);
int array_of_unsigned_int_into_AV(
        size_t *src,
        size_t src_sz,
        SV *dst
);
int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
);

int is_array_ref(
        SV *array,
        size_t *array_sz
){
        if( ! SvROK(array) ){ fprintf(stderr, "is_array_ref() : warning, input '%p' is not a reference.\n", array); return 0; }
        if( SvTYPE(SvRV(array)) != SVt_PVAV ){ fprintf(stderr, "is_array_ref() : warning, input ref '%p' is not an ARRAY reference.\n", array); return 0; }
        // it's an array, cast it to AV to get its len via av_len();
        // yes, av_len needs to be bumped up
        int asz = 1+av_len((AV *)SvRV(array));
        if( asz < 0 ){ fprintf(stderr, "is_array_ref() : error, input array ref '%p' has negative size!\n", array); return 0; }
        *array_sz = (size_t )asz;
        return 1; // success, it is an array and size returned by ref, above
}

#define array_numelts_1D(A,B) (!is_array_ref(A,B))

int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
){
        size_t anN, anN2, *Nd2 = NULL;

        if( ! is_array_ref(array, &anN) ){
           fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for array '%p'.\n", array);
           return 1;
        }

        if( *_Nd2 == NULL ){
           if( (Nd2=(size_t *)malloc(anN*sizeof(size_t))) == NULL ){
               fprintf(stderr, "array_numelts_2D() : error, failed to allocate %zu bytes for %zu items for Nd2.\n", anN*sizeof(size_t), anN);
               return 1;
           }
        } else Nd2 = *_Nd2;
        AV *anAV = (AV *)SvRV(array);
        size_t *pNd2 = &(Nd2[0]);
        for(size_t i=0;i<anN;i++,pNd2++){
           SV *subarray = *av_fetch(anAV, i, FALSE);
           if( ! is_array_ref(subarray, &anN2) ){
              fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for [%p][%p], item %zu.\n", array, subarray, i);
              if(*_Nd2==NULL) free(Nd2);
              return 1;
           }
           *pNd2 = anN2;
        }
        if( *_Nd2 == NULL ) *_Nd2 = Nd2;
        *_Nd1 = anN;
        return 0; // success
}

int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
){
        size_t dst_sz;
        if( ! is_array_ref(dst, &dst_sz) ){ fprintf(stderr, "array_of_int_into_AV() : error, call to is_array_ref() has failed.\n"); return 1; }
        AV *dstAV = (AV *)SvRV(dst);
        for(size_t i=0;i<src_sz;i++){
                av_push(dstAV, newSViv(src[i]));
        }
        return 0; // success
}
// end of Perl -> C -> Perl section
void print_2D_array(float *foo, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


node_t * head = NULL;
node_t * tail = NULL;

// these are where the initial input for each "feedforward" will be stored
float *host_x; 
float *device_x; 
float *host_x_transposed; 
float *device_x_transposed; 
SV    *perl_x; 
// these are where the target for each "feedforward" will be stored
float *host_y; 
float *device_y; 
float *host_y_transposed; 
float *device_y_transposed; 
SV    *perl_y; 

node* create_node(int insize, int outsize, SV *biases, SV *weights, int batch_size)
{
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    node_t * new_node = (node_t *)malloc(sizeof(node_t));
    new_node->input_size = insize;
    new_node->output_size = outsize;
    new_node->perl_Bias = biases;
    new_node->perl_Weights = weights;
// convert Perl bias array to C array of floats and push it onto the GPU
    new_node->device_Bias = gpu_device_malloc(sizeof(float)*outsize);
    new_node->host_Bias = gpu_host_malloc(sizeof(float)*outsize*1);

    array_numelts_2D(new_node->perl_Bias, &AH, &AWs);
    AW = AWs[0];

    pd = &(new_node->host_Bias[0]);
    av = (AV *)SvRV(new_node->perl_Bias);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
// convert Perl weight array to C array of floats and push it onto the GPU
    new_node->host_Weights = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights = gpu_device_malloc(sizeof(float)*insize*outsize);
    pd = &(new_node->host_Weights[0]);
    av = (AV *)SvRV(new_node->perl_Weights);
    for(i=0;i<outsize;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    gpu_memcpy_to_device(new_node->host_Bias, new_node->device_Bias, outsize*sizeof(float));
    gpu_memcpy_to_device(new_node->host_Weights, new_node->device_Weights, insize*outsize*sizeof(float));
// reserve memory for output and activated output (both 1 x outsize)
    new_node->host_Output = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Output = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output_Derivative = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output_Transposed = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output_Transposed = gpu_device_malloc(sizeof(float)*batch_size*outsize);
// reserve memory for deriviatives, bias = 1 * outsize, weight = insize * outsize
    new_node->host_Weights_Derivative = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights_Derivative = gpu_device_malloc(sizeof(float)*insize*outsize);
    new_node->host_Bias_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Bias_Derivative = gpu_device_malloc(sizeof(float)*batch_size*outsize);
// reserve memory for transposed Weights
    new_node->host_Weights_Transposed = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights_Transposed = gpu_device_malloc(sizeof(float)*insize*outsize);
// reserve memory for temporary derivative calculation
    new_node->host_Delta = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Delta = gpu_device_malloc(sizeof(float)*batch_size*outsize);
         
    new_node->next = NULL; 
    new_node->prev = NULL; 
    return new_node;
}        

int add_node(int insize, int outsize, SV *biases, SV *weights, int batch_size)
{

    node_t* new_node = create_node(insize, outsize, biases, weights, batch_size); 
    if (new_node == NULL) {
       std::cerr << "no node created for " << insize << " x " << outsize << std::endl;
       return 0;
    }
    if (tail == NULL) { 
        head = new_node; 
        tail = new_node; 
    } 
    else { 
        new_node->prev = tail; 
        tail->next = new_node; 
        tail = new_node; 
    } 
    return 1;
}

void reset_derivatives() {
    node_t * current = head;
    while (current != NULL) {
       for (int i = 0; i < current->output_size; ++i) {
           for (int j = 0; j < current->input_size; ++j) {
              current->host_Weights_Derivative[i *  current->input_size + j] = 0;
           }
           current->host_Bias_Derivative[i] = 0;
        }
        gpu_memcpy_to_device(current->host_Bias_Derivative, current->device_Bias_Derivative, current->output_size*sizeof(float));
        gpu_memcpy_to_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size * current->output_size * sizeof(float));
        current = current->next;
    }
}

int reserve_input_memory(int insize, int outsize, int batch_size)
{  
// memory for input array

    host_x = gpu_host_malloc(sizeof(float)*insize*batch_size); 
    host_x_transposed = gpu_host_malloc(sizeof(float)*insize*batch_size);
    device_x = gpu_device_malloc(sizeof(float)*insize*batch_size);
    device_x_transposed = gpu_device_malloc(sizeof(float)*insize*batch_size);

    host_y = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    host_y_transposed = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_y = gpu_device_malloc(sizeof(float)*outsize*batch_size);
    device_y_transposed = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    host_Cost_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_Cost_Derivative = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    host_Cost = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_Cost = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    mini_batch_size = batch_size;
    return 1;
}

int load_input(SV *x, int elements) 
{
// insize x 1 input array
    AV *av;
    float *pd;
    size_t i,j,insize;
    SV *subav, *subsubav; 
    mini_batch_size = elements;
    pd = &(host_x_transposed[0]);
    av = (AV *)SvRV(x);
    insize = head->input_size;

    for(i=0;i<elements;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_x_transposed, device_x_transposed, mini_batch_size*insize*sizeof(float));
    run_gpu_transpose_2D_array(device_x_transposed, device_x, mini_batch_size, insize);
    if (debug == 1) {
       gpu_memcpy_from_device(host_x, device_x, insize*mini_batch_size*sizeof(float));
       std::cout << "host_x"<<std::endl;
       print_2D_array(host_x, insize, mini_batch_size);
    }

    return 1;
}

int load_target(SV *y)
{
// outsize x 1 input array
    AV *av;
    float *pd; 
    size_t i,j,outsize;
    SV *subav, *subsubav; 
    pd = &(host_y_transposed[0]);
    av = (AV *)SvRV(y);
    outsize = tail->output_size;
    for(i=0;i<mini_batch_size;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<outsize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_y_transposed, device_y_transposed, mini_batch_size*outsize*sizeof(float));
    run_gpu_transpose_2D_array(device_y_transposed, device_y, mini_batch_size, outsize);
    if (debug == 1) {
printf("host_y %p device_y %p\n", host_y, device_y);
       gpu_memcpy_from_device(host_y, device_y, outsize*mini_batch_size*sizeof(float));
       std::cout << "host_y"<<std::endl;
       print_2D_array(host_y, outsize, mini_batch_size);
    }

    return 1;
}  


void run_feed_forward() {
    node_t * current = head;
    
    float *activation;
    activation = device_x;
    run_gpu_transpose_2D_array(activation, device_x_transposed, head->input_size,1);
    while (current != NULL) {
        run_gpu_linear( activation, current->device_Weights, current->device_Bias, current->device_Output, current->output_size, current->input_size, mini_batch_size );
        if (debug == 1) {
           if (current == head) {
              gpu_memcpy_from_device(host_x, device_x, head->input_size*mini_batch_size*sizeof(float));
              std::cout << "Initial Input Activations" << std::endl;
              print_2D_array(host_x, head->input_size, mini_batch_size);
           } else {
              gpu_memcpy_from_device(current->prev->host_Activated_Output, current->prev->device_Activated_Output, current->input_size*mini_batch_size*sizeof(float));
              std::cout << "Input Activations" << std::endl;
              print_2D_array(current->prev->host_Activated_Output, current->input_size, mini_batch_size);
           }
           gpu_memcpy_from_device(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float));
           std::cout << "Bias" << std::endl;
           print_2D_array(current->host_Bias, current->output_size, 1);
           gpu_memcpy_from_device(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float));
           std::cout << "Weights" << std::endl;
           print_2D_array(current->host_Weights, current->output_size, current->input_size);
           gpu_memcpy_from_device(current->host_Output, current->device_Output, mini_batch_size * current->output_size * sizeof(float));
           std::cout << "Output before activation" << std::endl;
           print_2D_array(current->host_Output, current->output_size, mini_batch_size);
        }

        gpu_sigmoid( current->device_Output, current->device_Activated_Output, current->output_size, mini_batch_size );
        if (debug == 1) {
           gpu_memcpy_from_device(current->host_Activated_Output, current->device_Activated_Output, mini_batch_size * current->output_size * sizeof(float));
           std::cout << "Output after activation" << std::endl;
           print_2D_array(current->host_Activated_Output,  current->output_size, mini_batch_size );
        }

        activation = current->device_Activated_Output;
        current = current->next;
    }
}

void run_backpropagation() {
   node_t *current = tail;

   //float *delta = device_Cost_Derivative;
   gpu_memcpy_intra_device( device_Cost_Derivative, current->device_Delta, mini_batch_size * current->output_size*sizeof(float));
   gpu_memcpy_intra_device( device_Cost_Derivative, current->device_Bias_Derivative, mini_batch_size * current->output_size*sizeof(float));
   if (debug == 1) {
      gpu_memcpy_from_device(current->prev->host_Activated_Output, current->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Activations " << std::endl;
      print_2D_array(current->prev->host_Activated_Output,  tail->input_size, mini_batch_size);
   }
   run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->prev->output_size,mini_batch_size);
   if (debug == 1) {
      gpu_memcpy_from_device(current->prev->host_Activated_Output_Transposed, current->prev->device_Activated_Output_Transposed, tail->input_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Activations Transposed" << std::endl;
      print_2D_array(current->prev->host_Activated_Output_Transposed, mini_batch_size, tail->input_size ); // it's transposed so now outsize x rows, rather than rows x outsize
      printf("host_Delta %p\n", current->host_Delta);
      printf("device_Delta %p\n", current->device_Delta);
      printf("outsize %d\n", tail->output_size);
      gpu_memcpy_from_device(current->host_Delta, current->device_Delta, tail->output_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Delta" << std::endl;
      print_2D_array(current->host_Delta, tail->output_size, mini_batch_size);
   }

   run_gpu_weight_derivative( current->device_Delta, current->prev->device_Activated_Output_Transposed, current->device_Weights_Derivative, current->output_size,  mini_batch_size,current->input_size ); 
   if (debug == 1) {
      gpu_memcpy_from_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size*current->output_size*sizeof(float));
      std::cout << "Last Layer Backpass Weights Derivative" << std::endl;
      print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
   }

   current = current->prev;
   while (current != NULL) {
// do the back prop
      run_gpu_sigmoid_prime(current->device_Activated_Output, current->device_Activated_Output_Derivative,current->output_size, mini_batch_size);
      if (debug == 1) {
         printf("host_Activated_Output_Derivative %p\n", current->host_Activated_Output_Derivative);
         printf("device_Activated_Output_Derivative %p\n", current->device_Activated_Output_Derivative);
         printf("outsize %d\n", current->output_size);
         gpu_memcpy_from_device(current->host_Activated_Output_Derivative, current->device_Activated_Output_Derivative, mini_batch_size*current->output_size*sizeof(float));
         std::cout << "Activated Output Derivative" << std::endl;
         print_2D_array(current->host_Activated_Output_Derivative, current->output_size, mini_batch_size);
      }

      float *sp = current->device_Activated_Output_Derivative;
      run_gpu_transpose_2D_array(current->next->device_Weights, current->next->device_Weights_Transposed, current->next->output_size, current->next->input_size);
      if (debug == 1) {
         gpu_memcpy_from_device(current->next->host_Weights_Transposed, current->next->device_Weights_Transposed, current->next->input_size*current->next->output_size*sizeof(float));
         std::cout << "Weights (next) transposed" << std::endl;
         print_2D_array(current->next->host_Weights_Transposed, current->next->input_size, current->next->output_size);
      }
      run_gpu_derivative(current->next->device_Weights_Transposed, current->next->device_Delta, sp, current->device_Delta, current->output_size, current->next->output_size,mini_batch_size); // wT x delta * sp (e.g. 30x10 x 10x1 => 30 x 1 * 30 x 1)
      if (debug == 1) {
         gpu_memcpy_from_device(current->host_Delta, current->device_Delta, mini_batch_size*current->output_size*sizeof(float));
         std::cout << "New delta" << std::endl;
         print_2D_array(current->host_Delta, current->output_size, mini_batch_size);
      }

      float *activation;
      if (current->prev != NULL) {
         run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->input_size,mini_batch_size);
         activation = current->prev->device_Activated_Output_Transposed;
      } else {
         activation = device_x_transposed;
      }
      gpu_memcpy_intra_device( current->device_Delta, current->device_Bias_Derivative, mini_batch_size * current->output_size*sizeof(float));
      run_gpu_matmul(current->device_Delta, activation, current->device_Weights_Derivative, current->output_size, mini_batch_size, current->input_size);
      if (debug == 1) {
         gpu_memcpy_from_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->output_size*current->input_size*sizeof(float));
         std::cout << "New Weights Derivative" << std::endl;
         print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
      }

      current = current->prev;
   }
}

void run_update_weights_and_biases(float modifier, float decay) {
   node_t * current = head;

   while (current != NULL) {
      run_gpu_update_weights(modifier, decay, current->device_Weights, current->device_Weights_Derivative, current->output_size, current->input_size);
      run_gpu_update_biases(modifier, current->device_Bias, current->device_Bias_Derivative, current->output_size, mini_batch_size);
      current = current->next;
   }
   dirty_dirty_weights = 1;
}

int get_last_activated_output( SV *R ) {
  
    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;
    
    RW = mini_batch_size;
    RH = tail->output_size;
   
// copy device data back to the host before loading the Perl values

    gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, RW*RH*sizeof(float));
   
    if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
    } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
    } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
    }
    
    pd = &(tail->host_Activated_Output[0]);
    av = (AV *)SvRV(R);
    for(i=0;i<RH;i++){ // for each row
        av2 = newAV(); // make a new array for each row
        av_extend(av2, RH); // extend it to hold #cols items (RW)
        // LeoNerd's suggestion
        av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
        for(j=0;j<RW;j++){ // for the cols of that row
            av_store(av2, j, newSVnv(*pd));
            pd++;
        }
    }
    if (debug == 1) {
       gpu_memcpy_from_device(tail->prev->host_Activated_Output, tail->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float));
       std::cout << "Input Activations" << std::endl;
       print_2D_array(tail->host_Activated_Output, mini_batch_size, tail->output_size);
       gpu_memcpy_from_device(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float));
       std::cout << "Bias" << std::endl;
       print_2D_array(tail->host_Bias, 1, tail->output_size);
       gpu_memcpy_from_device(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float));
       std::cout << "Weights" << std::endl;
       print_2D_array(tail->host_Weights, tail->output_size, tail->input_size);
       gpu_memcpy_from_device(tail->host_Output, tail->device_Output, RW*RH*sizeof(float));
       std::cout << "Output before activation" << std::endl;
       print_2D_array(tail->host_Output, RH, RW);
       std::cout << "Activtated Output" << std::endl;
       print_2D_array(tail->host_Activated_Output, RH, RW);
    }

    return 0;
}

float calculate_cost(){
      
   gpu_calculate_cost(tail->device_Activated_Output, device_y, device_Cost, tail->output_size, mini_batch_size, loss_function);

   if (debug == 1) { 
       gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float) );
       std::cout << "final layer activation" << std::endl;
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       std::cout << "targets" << std::endl;
       print_2D_array(host_y, tail->output_size, mini_batch_size);
   }
   gpu_memcpy_from_device(host_Cost, device_Cost, mini_batch_size * tail->output_size * sizeof(float) );
   float sum = 0;
   for (int i = 0; i < mini_batch_size; i++) {
      for (int j = 0; j < tail->output_size; j++) {
         sum += host_Cost[i * tail->output_size + j];
      }
   }
   if (debug == 1) {
       std::cout << "cost calc" << std::endl;
       print_2D_array(host_Cost, mini_batch_size, tail->output_size);
       std::cout << "sum of cost before weights calc " << sum << std::endl;
   }
   return sum;
}  

float calculate_weights_cost() {
   node_t * current = head;
   float sum = 0;
   while (current != NULL) {
      if (dirty_dirty_weights == 1) {
         gpu_memcpy_from_device(current->host_Weights, current->device_Weights, current->input_size * current->output_size*sizeof(float));
      }
      for (int i = 0; i < current->input_size; i++) {
         for (int j = 0; j < current->output_size; j++) {
            sum += powf(current->host_Weights[i * current->output_size + j],2);
         }
      }
      current = current->next;
   }
   dirty_dirty_weights = 0;
   return sum;
}     

void calculate_cost_derivative() {
   gpu_calculate_cost_and_derivative(tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, mini_batch_size, loss_function);
   if (debug == 1) {
       gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float));
       std::cout << "final layer activation" << std::endl;                      
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       gpu_memcpy_from_device(host_Cost_Derivative, device_Cost_Derivative, mini_batch_size * tail->output_size * sizeof(float));
       std::cout << "cost calc" << std::endl;
       print_2D_array(host_Cost_Derivative, mini_batch_size, tail->output_size);
   }
}

void set_debug_on() {
   debug = 1;
}

void set_debug_off() {
   debug = 0;
}

void set_loss(int funcno) { 
   loss_function = funcno;
}

void get_weights(SV *R, int i) {
   node_t * current = head;
   AV *av, *av2;
   float *pd;
   size_t j,RH,RW, asz;

   for (int skip = 0;skip < i; skip++) {
      current = current->next;
   }
   if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
   }

   RH = current->output_size;
   RW = current->input_size;
   gpu_memcpy_from_device(current->host_Weights, current->device_Weights, current->output_size*current->input_size*sizeof(float));

   pd = &(current->host_Weights[0]);
   av = (AV *)SvRV(R);
   for(i=0;i<RH;i++){ // for each row
       av2 = newAV(); // make a new array for each row
       av_extend(av2, RH); // extend it to hold #cols items (RW)
       // LeoNerd's suggestion
       av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
       for(j=0;j<RW;j++){ // for the cols of that row
           av_store(av2, j, newSVnv(*pd));
           pd++;
       }
   }
}

void get_biases(SV *R, int i) {
   node_t * current = head;
   AV *av, *av2;
   float *pd;
   size_t j,RH,RW, asz;

   for (int skip = 0;skip < i; skip++) {
      current = current->next;
   }
   if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
   }

   RH = current->output_size;
   RW = 1;
   gpu_memcpy_from_device(current->host_Bias, current->device_Bias, current->output_size*sizeof(float));

   pd = &(current->host_Bias[0]);
   av = (AV *)SvRV(R);
   for(i=0;i<RH;i++){ // for each row
       av2 = newAV(); // make a new array for each row
       av_extend(av2, RW); // extend it to hold #cols items (RW)
       // LeoNerd's suggestion
       av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
       for(j=0;j<RW;j++){ // for the cols of that row
           av_store(av2, j, newSVnv(*pd));
           pd++;
       }
   }
}

// PCA/Covariance data
float *host_Data, *host_Z, *host_Cov, *host_Means, *host_Stddev,
         *device_Data, *device_Z, *device_Cov, *device_Means, *device_Stddev
;
size_t CCH, CCW;

int calculate_covariance(SV *perl_Data, SV *perl_Cov) {
   // move Data to a C array
   size_t DH, DW, *DWs = NULL, // all of the arrays are the same size
          i, j, asz
        ;
   SV *subav, *subsubav, **ssubav;

   if( array_numelts_2D(perl_Data, &DH, &DWs) ){
      fprintf(stderr, "cuda_covariance() : error, call to array_numelts_2D() has failed for input matrix Data.\n");
      return 1;
   }

   CCH = DH; // CCH needed later for the final projection to the required number of columns

   DW = DWs[0];

   CCW = DW; // CCW needed later for the final projection to the required number of columns
   if (debug == 1) {
      std::cout << "CCH = " << CCH << " CCW " << CCW << std::endl;
   }
   if( is_array_ref(perl_Cov, &asz) ){
      if( asz > 0 ){
         AV *avd = (AV *)SvRV(perl_Cov);
         av_clear(avd);
      }
   } else if( SvROK(perl_Cov) ){
      // LeoNerd's suggestion:
      sv_setrv(SvRV(perl_Cov), (SV *)newAV());
   } else {
      // LeoNerd's suggestion:
      sv_setrv(perl_Cov, (SV *)newAV());
   }

   host_Data = gpu_host_malloc(sizeof(float)*DW*DH);
   host_Z = gpu_host_malloc(sizeof(float)*DW*DH);
   host_Cov = gpu_host_malloc(sizeof(float)*DW*DH);
   host_Means = gpu_host_malloc(sizeof(float)*DW*DH);
   host_Stddev = gpu_host_malloc(sizeof(float)*DW*DH);


   // allocate again for the device
   device_Data = gpu_device_malloc(sizeof(float)*DW*DH);
   device_Z = gpu_device_malloc(sizeof(float)*DW*DH);
   device_Cov = gpu_device_malloc(sizeof(float)*DW*DH);
   device_Means = gpu_device_malloc(sizeof(float)*DW*DH);
   device_Stddev = gpu_device_malloc(sizeof(float)*DW*DH);

   AV *av, *av2;
   float *pd = &(host_Data[0]);
   av = (AV *)SvRV(perl_Data);

   for(i=0;i<DH;i++){ // for each row
      ssubav = av_fetch(av, i, FALSE);
      if( ssubav == NULL ){
         fprintf(stderr, "cuda_covariance() : error, input matrix Data does not contain valid row at i=%d\n", i);
         return 1;
      }
      subav = *ssubav;
      for(j=0;j<DW;j++){ // for the cols of that row
         ssubav = av_fetch((AV *)SvRV(subav), j, FALSE);
         if( ssubav == NULL ){
            fprintf(stderr, "cuda_covariance() : error, input matrix Data does not contain valid column at i=%d, j=%d\n", i, j);
            return 1;
         }
         subsubav = *ssubav;
         *pd = SvNV(subsubav);
         pd++;
      }
   }
   // transfer results from host to device for A
   //print_2D_array(host_Data, DH, DW);
   gpu_memcpy_to_device(host_Data, device_Data, DW*DH*sizeof(float));

   for(i=0;i<DW;i++) { // initialise the means and stddev arrays to 0
      host_Means[i] = 0;
      host_Stddev[i] = 0;
   }

   gpu_memcpy_to_device(host_Means, device_Means, DW*sizeof(float));
   gpu_memcpy_to_device(host_Stddev, device_Stddev, DW*sizeof(float));

   run_gpu_calc_means(device_Data, device_Means, DH, DW);
   if (debug == 1) {
      gpu_memcpy_from_device(host_Means, device_Means, DW*sizeof(float));
      std::cout << "Means" << std::endl;
      print_2D_array(host_Means, 1, DW);
   }
   run_gpu_calc_stddev(device_Data, device_Means, device_Stddev, DH, DW);
   if (debug == 1) {
      gpu_memcpy_from_device(host_Stddev, device_Stddev, DW*sizeof(float));
      std::cout << "Stddev" << std::endl;
      print_2D_array(host_Stddev, 1, DW);
   }
   run_gpu_assign_z_scores(device_Data, device_Means, device_Stddev, device_Z, DH, DW);
   if (debug == 1) {
      gpu_memcpy_from_device(host_Z, device_Z, DH*DW*sizeof(float));
      std::cout << "Z" << std::endl;
      print_2D_array(host_Z, DH, DW);
   }
/* use centring by default - TODO normalisation should be done if requested 
   run_gpu_centre_data(device_Data, device_Means, device_Z, DH, DW);
   if (debug == 1) {
      gpu_memcpy_from_device(host_Z, device_Z, DH*DW*sizeof(float));
      std::cout << "Z" << std::endl;
      print_2D_array(host_Z, DH, DW);
   }
*/
   run_gpu_calc_covariance(device_Z, device_Cov, DH, DW);
   gpu_memcpy_from_device(host_Cov, device_Cov, DH*DW*sizeof(float));
   if (debug == 1) {
      std::cout << "Cov" << std::endl;
      print_2D_array(host_Cov, DW, DW);
   }
// populate the Perl array ref for Cov

   pd = &(host_Cov[0]);
   av = (AV *)SvRV(perl_Cov);
   av_extend(av, DH);
   for(i=0;i<DW;i++){ // for each row
      av2 = newAV(); // make a new array for each row
      av_extend(av2, DW); // extend it to hold #cols items (RW)
      // LeoNerd's suggestion
      av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
      for(j=0;j<DW;j++){ // for the cols of that row
         av_store(av2, j, newSVnv(*pd));
         pd++;
      }
   }

   //CUDA_CHECK(cudaFree((void *)device_Z));
   //CUDA_CHECK(cudaFree((void *)device_Cov));
   //CUDA_CHECK(cudaFreeHost((void *)host_Cov));
   //CUDA_CHECK(cudaFreeHost((void *)host_Z));
   //CUDA_CHECK(cudaFree((void *)device_Means));
   //CUDA_CHECK(cudaFreeHost((void *)host_Means));
   //CUDA_CHECK(cudaFree((void *)device_Stddev));
   //CUDA_CHECK(cudaFreeHost((void *)host_Stddev));

   return 0;
}

float *l2norm;
float *device_dotp;
float *device_R_T,
         *device_Sums, *host_Sums;
int cuda_qr_initialised = 0;

void cuda_qr_get_q_and_r(float *device_A, float *device_Q, float *device_R, size_t AH, size_t AW) {
   size_t i;
   if (cuda_qr_initialised == 0) {
      l2norm = gpu_device_malloc(sizeof(float)); 
      device_dotp = gpu_device_malloc(sizeof(float)*AW); 
      device_R_T = gpu_device_malloc(sizeof(float)*AW*AH); 
      cuda_qr_initialised = 1;
   }
   for (i=0;i<AW;i++) {
      run_gpu_qr_column_mult(device_A, device_Q, device_dotp, AH, AW, i);
      run_gpu_qr_column(device_A, device_Q, device_dotp, AH, AW, i);
      run_gpu_qr_l2_norm(device_Q, l2norm, AH, AW, i);
   }
   run_gpu_matmul( device_A, device_Q,  device_R_T, AH, AW, AW);
   run_gpu_transpose_2D_array(device_R_T, device_R,  AH, AW);
   run_gpu_qr_clamp_r_to_0(device_R, AH, AW); // make sure the cells below the diagonal are 0, not just close to it
}

void cuda_qr_get_q_and_r_cleanup() {
   gpu_free_device((void *)l2norm);
   gpu_free_device((void *)device_dotp);
   gpu_free_device((void *)device_R_T);

   cuda_qr_initialised = 0;
}

bool sums_less_than(int i, int j) { return (host_Sums[i] < host_Sums[j]); }
float *device_pQ; // we'll need this for the projection later on

int cuda_eigenvectors( SV *perl_pQ, float epsilon, int max_iterations) ;
int cuda_eigenvectors( SV *perl_pQ, float epsilon, int max_iterations) {
// relies on host_Cov and device_Cov being already populated by calculate_covariance
   size_t pQH, pQW, *pQWs = NULL,
          XH, XW,
          RH, RW, QH, QW,
          i, j, asz
        ;
   float *host_pQ,
         *device_pQ2, 
         *device_Q, *device_R
        ;
   SV *subav, *subsubav, **ssubav;
   int *sums_indicies, *device_sums_indicies;

   if( array_numelts_2D(perl_pQ, &pQH, &pQWs) ){
      fprintf(stderr, "cuda_qr_get_q() : error, call to array_numelts_2D() has failed for input matrix pQ.\n");
      return 1;
   }

   pQW = pQWs[0];

   XH = pQH; XW = pQW;

   if (debug == 1) {
      printf("cuda_eigenvectors incoming pQ size = %d rows x %d columns\n", pQH, pQW);
   }
   host_pQ = gpu_host_malloc(sizeof(float)*pQW*pQH);

   AV *av, *av2;
   float *pd = &(host_pQ[0]);
   av = (AV *)SvRV(perl_pQ);

   for(i=0;i<pQH;i++){ // for each row
      ssubav = av_fetch(av, i, FALSE);
      if( ssubav == NULL ){
         fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix pQ does not contain valid row at i=%d\n", i);
         return 1;
      }
      subav = *ssubav;
      for(j=0;j<pQW;j++){ // for the cols of that row
         ssubav = av_fetch((AV *)SvRV(subav), j, FALSE);
         if( ssubav == NULL ){
            fprintf(stderr, "inline_cuda_matrix_multiply() : error, input matrix pQ does not contain valid column at i=%d, j=%d\n", i, j);
            return 1;
         }
         subsubav = *ssubav;
         *pd = SvNV(subsubav);
         pd++;
      }
   }

   RH = pQH; RW = pQW;
   QH = pQH; QW = pQW;

// initialise working area for the eigenvector calc

   device_pQ = gpu_device_malloc(sizeof(float)*pQW*pQH);
   device_pQ2 = gpu_device_malloc(sizeof(float)*pQW*pQH);
   device_R = gpu_device_malloc(sizeof(float)*pQW*pQH);
   device_Q = gpu_device_malloc(sizeof(float)*pQW*pQH);
   device_Sums = gpu_device_malloc(sizeof(float)*pQW);
   host_Sums = gpu_host_malloc(sizeof(float)*pQW);
   float *host_Q = gpu_host_malloc(sizeof(float)*pQW*pQH);
   float *host_R = gpu_host_malloc(sizeof(float)*pQW*pQH);

   gpu_memcpy_to_device(host_pQ, device_pQ, pQW*pQH*sizeof(float));

   int iterations = 0;
   int host_unconverged = 1;
   while (host_unconverged == 1 && iterations++ < max_iterations) {
      cuda_qr_get_q_and_r(device_Cov, device_Q, device_R, XH, XW);
      run_gpu_matmul( device_pQ, device_Q,  device_pQ2, pQH, pQW, pQW);
      // I guess we could do something smart here to avoid the memcpy?
      gpu_memcpy_intra_device( device_pQ2, device_pQ, pQH*pQW*sizeof(float));
      gpu_reset_unconverged();
      run_gpu_matmul_check_converged( device_R, device_Q,  device_Cov, epsilon, XH, XW, XW);
      host_unconverged = gpu_get_unconverged_state();
   }
   if (debug == 1) {
      std::cout << "converged in " << iterations << " iterations" << std::endl;
      gpu_memcpy_from_device(host_Cov, device_Cov, pQH*pQW*sizeof(float));
      std::cout << "Eigen values"<<std::endl;
      print_2D_array(host_Cov, pQH, pQW);
   }
// at this point the basic eigenvectors are calculated, so we need to sum each one
// and if the sum < 1, multiply the vector by -1
   run_gpu_eigenvector_signs(device_pQ, device_Sums, pQH, pQW);
   gpu_memcpy_from_device(host_Sums, device_Sums, pQW*sizeof(float));
   
   sums_indicies = (int*)malloc(pQW * sizeof(int));
   for (int i=0;i<pQW;i++) {
      sums_indicies[i] = i;
   }
   std::vector<int> sums_vector(sums_indicies, sums_indicies+pQW);
   std::sort(sums_vector.begin(), sums_vector.end(), sums_less_than);
   i = 0;
   for (std::vector<int>::iterator it=sums_vector.begin();it!=sums_vector.end();++it)
       sums_indicies[i++] = *it;
   // create a new matrix with the sorted eigenvectors
   // might as well reuse pQ2, since is already allocated... 
   // first copy the array with the new order to the device
   device_sums_indicies = gpu_device_malloc_int( sizeof(int)*pQW);
   gpu_memcpy_to_device_int(sums_indicies, device_sums_indicies, pQW*sizeof(int));
   run_gpu_reorder_eigenvectors(device_pQ2, device_pQ, device_sums_indicies, pQH, pQW);
   gpu_memcpy_intra_device( device_pQ2, device_pQ, pQH*pQW*sizeof(float));

   gpu_free_device((void *)device_R);
   gpu_free_device((void *)device_Q);

   // the eigenvectors are in pQ, so copy them back to Perl
   // transfer results from device to host
   gpu_memcpy_from_device(host_pQ, device_pQ, pQW*pQH*sizeof(float));
   if (debug == 1) {
      std::cout << "Eigenvectors Post Signs"<<std::endl;
      print_2D_array(host_pQ, pQH, pQW);
   }

   // clear the existing pQ Perl data structure, as it gets added to rather than overwritten
   if( is_array_ref(perl_pQ, &asz) ){
      if( asz > 0 ){
         AV *avd = (AV *)SvRV(perl_pQ);
         av_clear(avd);
      }
   } else if( SvROK(perl_pQ) ){
      // LeoNerd's suggestion:
      sv_setrv(SvRV(perl_pQ), (SV *)newAV());
   } else {
      // LeoNerd's suggestion:
      sv_setrv(perl_pQ, (SV *)newAV());
   }

   pd = &(host_pQ[0]);
   av = (AV *)SvRV(perl_pQ);
   av_extend(av, pQH);
   for(i=0;i<pQH;i++){ // for each row
      av2 = newAV(); // make a new array for each row
      av_extend(av2, pQW); // extend it to hold #cols items (RW)
      // LeoNerd's suggestion
      av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
      for(j=0;j<pQW;j++){ // for the cols of that row
         av_store(av2, j, newSVnv(*pd));
         pd++;
      }
   }

//   CUDA_CHECK(cudaFree((void *)device_pQ));
//   CUDA_CHECK(cudaFree((void *)device_pQ2));
//   CUDA_CHECK(cudaFreeHost((void *)host_pQ));
//   CUDA_CHECK(cudaFree((void *)device_X));
//   CUDA_CHECK(cudaFreeHost((void *)host_X));
   return 0;
}

int cuda_project_results(int projected_columns, SV *perl_projection){
        size_t pH, pW, // projection
// Z, which is what was are multiplying the results by, was already calculated in the PCA code
               i, j, asz
        ;
        float  *host_p, *device_p ;

        SV *subav, *subsubav, **ssubav;

        pH = CCH; // assumes calculate_covariance already called and CCH populated
        pW = projected_columns;

        if( is_array_ref(perl_projection, &asz) ){
           if( asz > 0 ){
              AV *avd = (AV *)SvRV(perl_projection);
              av_clear(avd);
           }
        } else if( SvROK(perl_projection) ){
           // LeoNerd's suggestion:
           sv_setrv(SvRV(perl_projection), (SV *)newAV());
        } else {
           // LeoNerd's suggestion:
           sv_setrv(perl_projection, (SV *)newAV());
        }

        host_p = gpu_host_malloc(sizeof(float)*pW*pH);
        device_p = gpu_device_malloc(sizeof(float)*pW*pH);
        run_gpu_partial_matmul( device_Z, device_pQ, device_p, CCH, CCW, CCW, projected_columns);

        // free A and B from Host and device
        //CUDA_CHECK(cudaFreeHost(host_Results));
        //CUDA_CHECK(cudaFreeHost(host_B));
        //CUDA_CHECK(cudaFree((void *)device_A));
        //CUDA_CHECK(cudaFree((void *)device_B));

        // transfer results from device to host
        gpu_memcpy_from_device(host_p, device_p, pH*pW*sizeof(float));


        float *pd = &(host_p[0]);
        AV *av = (AV *)SvRV(perl_projection);
        AV *av2;
        av_extend(av, pH);
        for(i=0;i<pH;i++){ // for each row
           av2 = newAV(); // make a new array for each row
           av_extend(av2, pW); // extend it to hold #cols items (RW)
           // LeoNerd's suggestion
           av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
           for(j=0;j<pW;j++){ // for the cols of that row
              av_store(av2, j, newSVnv(*pd));
              pd++;
           }
        }

        //CUDA_CHECK(cudaFree((void *)device_R));
        //CUDA_CHECK(cudaFreeHost((void *)host_R));

        return 0;
}

