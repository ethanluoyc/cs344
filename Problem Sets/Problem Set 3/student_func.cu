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
#include "loadSaveImage.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_kernel(float* const d_out, float* d_in,
                        const size_t logLumSize,
                        bool isMax) {
  int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
  int tid = threadIdx.x;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (isMax) {
        // Maximum
        d_in[myId] = max(d_in[myId], d_in[myId + s]);
      } else {
        // Minimum
        d_in[myId] = min(d_in[myId], d_in[myId + s]);
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }

}


void reduce(float &id,
            const float* const d_logLuminance, const size_t logLumSize,
            bool isMax) {
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int blocks = logLumSize / maxThreadsPerBlock;

  float *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * logLumSize));
  checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, sizeof(float) * logLumSize, cudaMemcpyHostToDevice));

  float *d_intermediate;
  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * logLumSize));
  checkCudaErrors(cudaMemcpy(d_intermediate, d_logLuminance, sizeof(float) * logLumSize, cudaMemcpyHostToDevice));

  float *d_id;
  checkCudaErrors(cudaMalloc(&d_id, sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_id, &id, sizeof(float), cudaMemcpyHostToDevice));


  // First step
  int blockSize = blocks;
  int threadSize = threads;
  reduce_kernel<<<blockSize, threadSize>>>(d_intermediate, d_in, logLumSize, isMax);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Second step
  threadSize = blocks;
  blockSize = 1;
  reduce_kernel<<<blockSize, threadSize>>>(d_id, d_intermediate, logLumSize, isMax);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&id, d_id, sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__ void histogram_kernel(const float* const d_logLuminance, const size_t logLumSize,
                                 int* d_bins, float lumMin, float lumRange, int numBins) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < logLumSize) {
    int bin = (int) ((d_logLuminance[idx] - lumMin) / lumRange * numBins);
    if (bin == numBins) bin--;
    atomicAdd(&d_bins[bin], 1);
  }
}

// Hillis Steele scanl implementation of cdf
__global__ void cdf_kernel(int* d_bins, unsigned int* d_cdf,
                           int logLumSize, int numBins) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int d = 0; d < (int) ceil(log2((float) numBins)); d++) {
    int _k = 2>>(d-1); 
    if (idx < numBins && idx >= _k) {
      d_bins[idx] = d_bins[idx - _k<<1] + d_bins[idx];
    }
  }
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
  int logLumSize = numCols * numRows;

  // 1) find the minimum and maximum value in the input logLuminance channel
  // store in min_logLum and max_logLum
  reduce(max_logLum, d_logLuminance, logLumSize, true);
  reduce(min_logLum, d_logLuminance, logLumSize, false);

  // 2) subtract them to find the range
  float range = max_logLum - min_logLum;
  printf("max: %f, min %f, range: %f\n", max_logLum, min_logLum, range);

  /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  int *d_bins;
  checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
  checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));
  histogram_kernel<<<numRows, numCols>>>(d_logLuminance, logLumSize, d_bins, min_logLum, range, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(d_cdf, d_bins, sizeof(float) * numBins, cudaMemcpyHostToDevice));
  cdf_kernel<<<1, numBins>>>(d_bins, d_cdf, logLumSize, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
