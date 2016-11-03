/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"
#include "cstdio"
#include "math.h"
#include "stdbool.h"

#include "thrust/device_vector.h"
#include "thrust/extrema.h"
#include "thrust/host_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/fill.h"
#include "thrust/sort.h"
#include "thrust/scan.h"
#include "thrust/reduce.h"
#include "thrust/sequence.h"
#include "thrust/set_operations.h"
#include <thrust/merge.h>
#include "thrust/functional.h"
#include <thrust/pair.h>
#include <thrust/tuple.h>



// Wrapper code for time function
#define time(X,Y) \
	do { \
		timer.Start(); \
		X; \
		if (Y) display_maxmin(&min_logLum, &max_logLum); \
		timer.Stop(); \
		printf("Your code executed in %g ms\n\n", timer.Elapsed()); \
	} while (0);

	
struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

// implemantion of thrust libraries max/min
void thrust_maxmin(	const float* const  d_logLuminance,
					const int array_size,
					float* min_logLum,
					float* max_logLum){
						
	printf("Thrust library implementation\n");
	thrust::device_ptr<const float> dev_ptr = thrust::device_pointer_cast(d_logLuminance);
	thrust::device_ptr<const float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + array_size);
    thrust::device_ptr<const float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + array_size);

	*min_logLum = min_ptr[0];
	*max_logLum = max_ptr[0];
}

// naive serial implemantion of max/min
void serial_maxmin(	const float* const  d_logLuminance,
					const int array_size,
					float* min_logLum,
					float* max_logLum){
	printf("Serial implementation\n");
	const int array_bytes = array_size * sizeof(float);
	float *  h_logLuminance = (float *) malloc(array_bytes);
	cudaMemcpy( h_logLuminance , d_logLuminance, array_bytes, cudaMemcpyDeviceToHost);
	
	*min_logLum = 10;
	*max_logLum = -10;
	for (int i = 0; i < array_size; i++){
		if (h_logLuminance[i] < *min_logLum){
			*min_logLum = h_logLuminance[i];
		}
		if (h_logLuminance[i] > *max_logLum){
			*max_logLum = h_logLuminance[i];
			}
	}
	free(h_logLuminance);
}

// shared memory version of max kernel
__global__ void shmem_max(	float * d_out,
							const float * d_in,
							int threads,
							int padded_threads)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
	if (tid < threads){
		sdata[tid] = d_in[myId];
	}
	// padding in data to fill power of 2 requirement
	else {
		sdata[tid] = d_in[0];
	}
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
	// this generates exponential values of 2 (e.g. 16, 8, 4, 2 , 1)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid],sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

// shared memory version of min kernel
__global__ void shmem_min(	float * d_out,
							const float * d_in,
							int threads,
							int padded_threads)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
	if (tid < threads){
		sdata[tid] = d_in[myId];
	}
	// padding in data to fill power of 2 requirement
	else {
		sdata[tid] = d_in[0];
	}
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
	// this generates exponential values of 2 (e.g. 16, 8, 4, 2 , 1)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = min(sdata[tid],sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

// serial wrapper code for gpu maxmin
void gpu_maxmin(const float* const  d_logLuminance,
				const int array_size,
				float* min_logLum,
				float* max_logLum){
	printf("Shared memory implentation\n");
	const int array_bytes = array_size * sizeof(float);
    float * d_intermediate, * d_out;

	cudaMalloc((void **) &d_intermediate, array_bytes); // overallocated
    cudaMalloc((void **) &d_out, sizeof(float));

	 // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = array_size / maxThreadsPerBlock;
    shmem_max<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_logLuminance, maxThreadsPerBlock , maxThreadsPerBlock);
			
    // now we're down to one block left, so reduce it
	// the number of threads needs to be a power of 2
	int padded_threads = pow(2,ceil(log2( (double) blocks)));
    threads = padded_threads; // launch one thread for each block in prev step
    blocks = 1;
    shmem_max<<<blocks, threads, padded_threads * sizeof(float)>>>
            (d_out, d_intermediate , threads, padded_threads);
    
	cudaMemcpy(max_logLum , d_out, sizeof(float), cudaMemcpyDeviceToHost);
	
	threads = maxThreadsPerBlock;
    blocks = array_size / maxThreadsPerBlock;
    shmem_min<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_logLuminance, maxThreadsPerBlock , maxThreadsPerBlock);
			
    // now we're down to one block left, so reduce it
	// the number of threads needs to be a power of 2
    threads = padded_threads; // launch one thread for each block in prev step
    blocks = 1;
    shmem_min<<<blocks, threads, padded_threads * sizeof(float)>>>
            (d_out, d_intermediate , threads, padded_threads);
    
	cudaMemcpy(min_logLum , d_out, sizeof(float), cudaMemcpyDeviceToHost);

	// clean up
	cudaFree(d_intermediate);
	cudaFree(d_out);
}

void display_maxmin(float* min_logLum,
                    float* max_logLum){
	printf("Maximum: %f\n" , *max_logLum);
	printf("Mininum: %f\n" , *min_logLum);
}

// atomic (naive) implemantion of historgram
__global__ void atomic_histogram(unsigned int * d_bins ,
									const float * d_logLuminance,
									size_t numBins ,
									float lumRange ,
									float min_logLum){
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	float myItem = d_logLuminance[myId];
	// the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	int myBin = round(((myItem - min_logLum) / lumRange) * (float) numBins);
	atomicAdd(&(d_bins[myBin]) , 1 );
}

// (slow) atomic version of pdf to cdf
__global__ void atomic_cdf(unsigned int * d_pdf , unsigned int* const d_cdf , int numBins){
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int myItem = d_pdf[myId];
	for (int i = myId ; i < numBins ; i++){
		atomicAdd( &(d_cdf[i]) , myItem);
	}
}

// sets array to 0
__global__ void set2zero(unsigned int * d_pdf){
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	d_pdf[myId] = 0;
}

// Serial wrapper for atomic historgram implemantion
void atomic_histogram(	const float* const d_logLuminance,
							unsigned int* const d_cdf,
							const size_t numBins,
							const int array_size,
							float lumRange,
							float min_logLum){
	
	printf("Atomic histogram\n");
	unsigned int * d_pdf;
    cudaMalloc((void **) &d_pdf, sizeof(int) * numBins);
	int threads = numBins;
	int blocks = 1;
	set2zero<<<blocks, threads>>>(d_pdf);

	blocks = array_size / 1024;
	threads = 1024; // max_threads
	atomic_histogram<<<blocks, threads>>>(d_pdf , d_logLuminance, numBins , lumRange , min_logLum);
	atomic_cdf<<<blocks, threads>>>(d_pdf , d_cdf , numBins);

	cudaFree(d_pdf);

}


__global__ void lum2bin(	unsigned int * d_pdf, 
							const float* const  d_logLuminance,
							float C,
							float A){
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	d_pdf[myId] = round(A * d_logLuminance[myId] + C);
}

__global__ void vector2array(	unsigned int* const d_cdf_pointer,
								unsigned int* const d_cdf){
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	d_cdf[myId] = d_cdf_pointer[myId];
}

void thrust_histogram(	const float* const d_logLuminance,
						unsigned int* const d_cdf,
						const size_t numBins,
						const int array_size,
						float lumRange,
						float min_logLum){
	printf("Thrust histogram\n");
	
	// Constants for SAXPY (Single-Precision A·X Plus Y)
	float C = - min_logLum / lumRange * (float) numBins;
	float A = (float) numBins / lumRange;
	unsigned int * d_pdf;
    cudaMalloc((void **) &d_pdf, sizeof(unsigned int) * numBins);
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
    int blocks = array_size / maxThreadsPerBlock;
	// mapping to bin
	//thrust::device_vector<unsigned int> key_vec(array_size);                    
	lum2bin<<<blocks, threads>>>(d_pdf , d_logLuminance, C , A);
	
	// Use thrust library to sort by bin
	// copy data into thrust vector
    thrust::device_ptr<unsigned int> d_ptr(d_pdf);
	thrust::device_vector<unsigned int> key_vec(d_ptr , d_ptr +  array_size);
 
    //thrust::exclusive_scan(dev_ptr, dev_ptr+array_size, key_vec.begin());
	
	// sort vector
	//thrust::sort(key_vec.begin(), key_vec.end());
	// key_vec is size of array that looks like 000111122244455555
	
/* 	// fill vector with 1's and 0's
	thrust::device_vector<unsigned int> values_vec(array_size * 2);
    thrust::fill(values_vec.begin(), values_vec.begin() + array_size, 1);
	thrust::fill(values_vec.begin() + array_size, values_vec.end(), 0);

	// create a vector of 0's that is the length of empty bins
	// I need a list of elements that are not in bin
	// 0..numBins
	
	thrust::device_vector<unsigned int> empty_key_vec(array_size);
	thrust::sequence(empty_key_vec.begin(), empty_key_vec.end());

	key_vec.insert(key_vec.end(),empty_key_vec.begin(), empty_key_vec.end());

	// join keys
	// sort again
	
	// reduce by keys
	thrust::reduce_by_key(key_vec.begin(), key_vec.end(), values_vec.begin(), key_vec.begin(), values_vec.begin());
	


	
	
	// Convert pdf to cdf
	thrust::device_vector<unsigned int> d_cdf_vector(values_vec.size()); 
	
	thrust::exclusive_scan(values_vec.begin(), values_vec.end(), d_cdf_vector.begin());  */
	unsigned int* d_cdf_pointer;
	d_cdf_pointer = (unsigned int*) thrust::raw_pointer_cast(&key_vec[0]);
	//d_cdf_pointer = (unsigned int*) thrust::raw_pointer_cast(&d_cdf_vector[0]);
	// copy data into d_cdf	
	vector2array<<<blocks, threads>>>(d_cdf_pointer , d_cdf);
	cudaFree(d_pdf);
	
}
void printCDF(unsigned int* const d_cdf , const size_t numBins){
  thrust::device_ptr<unsigned int> dev_ptr_key        = thrust::device_pointer_cast(d_cdf);

	//unsigned int* const h_cdf = (unsigned int* const) malloc(numBins * sizeof(unsigned int));
	//cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int* const) , cudaMemcpyDeviceToHost);
	for (int i = 0; i < numBins ; i++){
		printf("%i: %i\n",i,(unsigned int)*(dev_ptr_key+i));
		//printf("%i: %i\n" , i , h_cdf[i]);
	};
	//free(h_cdf);
}

// maps 1024 elements to the indvidual bin
__global__ void gpu_histogram_ker(unsigned int * d_pdf, unsigned int * d_local_histo , unsigned int numThread){
	int id = threadIdx.x;
	int his_id = blockDim.x * id;
	int pdf_id = id * numThread;
	for (int i = 0; i < numThread ; i++){
		int bin = d_pdf[pdf_id + i];
		// increase local histogram in bin by 1
		d_local_histo[his_id + bin]++; 
	}
}

// This code is inneffiecent for two reasons. The for loop is not utilizing the full extent of the gpu
// The code is reading from non-sequential areas.
__global__ void sum_local_histo(unsigned int * d_local_histo , unsigned int* const d_cdf, unsigned int numThread){
	int id = threadIdx.x;
	for (int i = 0; i < numThread ; i++){
		d_cdf[id] = d_cdf[id] + d_local_histo[id + i * numThread];
	}
	
}

__global__ void hillissteel_scan(unsigned int* const d_cdf){
		int id = threadIdx.x;
		// this generates exponential values of 2 (e.g. 16, 8, 4, 2 , 1)
		// I want 1, 2,4,8
		for (unsigned int i = 1; i < blockDim.x ; i <<= 1){
			int neighbor = id - i;
			if (neighbor > 0) d_cdf[id] = d_cdf[neighbor] + d_cdf[id];
			__syncthreads();            // make sure entire block is loaded!
		}
}


void gpu_histogram(	const float* const d_logLuminance,
							unsigned int* const d_cdf,
							const size_t numBins,
							const int array_size,
							float lumRange,
							float min_logLum){
	
	printf("Gpu histogram\n");
	const int maxThreadsPerBlock = 1024;

	
	unsigned int * d_local_histo;
	cudaMalloc((void **) &d_local_histo, sizeof(unsigned int) * numBins * maxThreadsPerBlock);
	int threads = 1024; // max_threads
	int blocks = numBins;
	set2zero<<<blocks, threads>>>(d_local_histo);
	
    blocks = 1;
	threads = numBins;
	set2zero<<<blocks, threads>>>(d_cdf);
	
	
	// Constants for SAXPY (Single-Precision A·X Plus Y)
	float C = - min_logLum / lumRange * (float) numBins;
	float A = (float) numBins / lumRange;
	unsigned int * d_pdf;
    cudaMalloc((void **) &d_pdf, sizeof(unsigned int) * array_size);
	threads = maxThreadsPerBlock;
    blocks = array_size/ threads;
	lum2bin<<<blocks, threads>>>(d_pdf , d_logLuminance, C , A);
	// d_pdf is an unsorted listing of a 1to1 transformation from luminance to bin


	// launch a kernel that computes a local histogram of 1024 elements

	unsigned int numThread = array_size / threads;
	assert(array_size % threads == 0);
	blocks = 1;
	threads = 1024; // max_threads
	gpu_histogram_ker<<<blocks, threads>>>(d_pdf, d_local_histo,numThread);

	threads = maxThreadsPerBlock;
    blocks = numBins/ threads;
	set2zero<<<blocks, threads>>>(d_pdf);

	// now I need to sum all the local histograms 
	numThread = 1024;
	blocks = blocks;
	threads = threads;
	sum_local_histo<<<blocks, threads>>>(d_local_histo , d_cdf , numThread);
			
	blocks = 1;
	threads = numBins; // max_threads
	hillissteel_scan<<<blocks, threads>>>(d_cdf);
	
	cudaFree(d_local_histo);
	cudaFree(d_pdf); 

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumul vfgmative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	// declare GPU memory pointers
    GpuTimer timer;
	const int array_size = numCols * numRows;
	
	// Three different implemantions of max,min function
	time(thrust_maxmin(d_logLuminance , array_size , &min_logLum , &max_logLum), true)
	time(serial_maxmin(d_logLuminance , array_size , &min_logLum , &max_logLum), true)
	time(   gpu_maxmin(d_logLuminance , array_size , &min_logLum , &max_logLum), true)

	float lumRange = max_logLum - min_logLum;
	
	// Two different implemantions of histogram
	time(atomic_histogram(d_logLuminance , d_cdf , numBins  ,array_size , lumRange , min_logLum) , false);
	//time(thrust_histogram(d_logLuminance , d_cdf,numBins,array_size,lumRange, min_logLum),false);
	time(gpu_histogram(d_logLuminance , d_cdf , numBins  ,array_size , lumRange , min_logLum) , false);

	
}
