#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>       /* fabsf */
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 0

//Error check-----
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//Error check-----
//This is a very good idea to wrap your calls with that function.. Otherwise you will not be able to see what is the error.
//Moreover, you may also want to look at how to use cuda-memcheck and cuda-gdb for debugging.

__global__ void parallel_cycles(int* d_xadj,int* d_adj, int* d_nv, int* d_result){
  
  //TO DO: GPU SCALE
  printf("Number of vertices: %d \n", *d_nv );
  
  
}

void wrapper(int* xadj, int* adj,int* no_vertices, int k){
  
  printf("Wrapper here! \n");


  int no_thread = 8;

  int* d_xadj;
  int* d_adj;
  int* d_result;
  int* h_result;
  int* d_nv;
 
  h_result = (int*)malloc(sizeof(int));
  *h_result = 0;



  //TO DO: DRIVER CODE
  cudaSetDevice(0);

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaMalloc( (void **) &d_xadj, *xadj *sizeof(int));
  cudaMalloc( (void **) &d_adj, *adj *sizeof(int));
  cudaMalloc( (void **) &d_result, sizeof(int));
  cudaMalloc( (void **) &d_nv, sizeof(int));


  cudaMemcpy(d_xadj, xadj, (*no_vertices) * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_adj, adj, (*no_vertices) * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_result, h_result, sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_nv, no_vertices, sizeof(int), cudaMemcpyHostToDevice );

  int no_blocks = (ceil)((*no_vertices)/no_thread);
  
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  
  parallel_cycles<<<1,no_thread>>>(d_xadj, d_adj, d_nv, d_result);
  cudaDeviceSynchronize(); 
  gpuErrchk( cudaDeviceSynchronize() );
  
  
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU scale took: %f s\n", elapsedTime/1000);
  
    
}