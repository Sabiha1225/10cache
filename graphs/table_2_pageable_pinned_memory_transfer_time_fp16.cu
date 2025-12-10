#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h> // For __half type and utilities

// Convenience function for checking CUDA runtime API results
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileCopies(__half       *h_a, 
                   __half       *h_b, 
                   __half       *d, 
                   unsigned int  n,
                   char         *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(__half);

  // events for timing
  cudaEvent_t startEvent, stopEvent; 

  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // Host to Device
  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  float time;
  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("Transfer Time, %f ms\n", time);
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  // Device to Host
  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("Transfer Time, %f ms\n", time);
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (__half2float(h_a[i]) != __half2float(h_b[i])) { // Convert for comparison
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
}

int main()
{
  unsigned int nElements = 4 * 1024 * 1024;
  const unsigned int bytes = nElements * sizeof(__half);

  // host arrays
  __half *h_aPageable, *h_bPageable;   
  __half *h_aPinned, *h_bPinned;

  // device array
  __half *d_a;

  // allocate and initialize
  h_aPageable = (__half*)malloc(bytes);                    // host pageable
  h_bPageable = (__half*)malloc(bytes);                    // host pageable
  checkCuda(cudaMallocHost((void**)&h_aPinned, bytes));    // host pinned
  checkCuda(cudaMallocHost((void**)&h_bPinned, bytes));    // host pinned
  checkCuda(cudaMalloc((void**)&d_a, bytes));              // device

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = __float2half(i);      
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

  printf("\n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}
