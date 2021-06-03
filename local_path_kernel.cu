#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Error check-----
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
// Error check-----
// This is a very good idea to wrap your calls with that function.. Otherwise
// you will not b Moreover, you may also want to look at how to use
// cuda-memcheck and cuda-gdb for debuggin

__device__ bool device_contains(int *array, int start, int end, int item) {
  for (int j = start; j < end; j++) {
    if (array[j] == item)
      return true;
  }
  return false;
}
__device__ void deviceDFS(int *xadj, int *adj, int *nov, int k, int max_k,
                          int vertex, int *counter, int start, int *my_path) {
  // printf("DFS on %d at k %d\n",vertex,k);
  my_path[max_k - k - 1] = vertex;
  // for(int i=0; i<max_k-k;i++)
  // printf("path element %d is %d\n",i,my_path[i]);
  if (k == 0) {
    if (device_contains(adj, xadj[vertex], xadj[vertex + 1], start))
      atomicAdd(&counter[start], 1);
    return;
  }
  // printf("my marked is at%p\n",(void *) marked);
  for (int j = xadj[vertex]; j < xadj[vertex + 1]; j++) {
    // if (!device_contains(my_path, 0, max_k - k, adj[j])) {
      deviceDFS(xadj, adj, nov, k - 1, max_k, adj[j], counter, start, my_path);
    }
  }
}
__global__ void prep(int *xadj, int *adj, int *nov, int k, int max_k, int *ct) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < *nov) {
    // int* my_path= new int[k];
    int my_path[6];
    deviceDFS(xadj, adj, nov, k - 1, max_k, id, ct, id, my_path);
  }
}
__global__ void setct(int *nov, int *ct) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < *nov) {
    ct[id] = 0;
  }
}
void wrapper(int *xadj, int *adj, int *nov, int nnz, int k) {
  cudaSetDevice(0);
  int *d_xadj;
  int *d_adj;
  int *d_nov;
  int *d_ct;
  int *ct = new int[*nov];
  cudaMalloc((void **)&d_xadj, (*nov + 1) * sizeof(int));
  cudaMalloc((void **)&d_adj, nnz * sizeof(int));
  cudaMalloc((void **)&d_nov, sizeof(int));
  cudaMalloc((void **)&d_ct, (*nov) * sizeof(int));

  cudaMemcpy(d_xadj, xadj, (*nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_adj, adj, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nov, nov, sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned int threads = prop.maxThreadsPerBlock;
  std::cout << "Device Properties" << std::endl;
  std::cout << "The threads: " << threads << std::endl;
  gpuErrchk(cudaDeviceSynchronize());
#ifdef DEBUG
  std::cout << "malloc copy done" << std::endl;
#endif
  setct<<<(*nov + threads - 1) / threads, threads>>>(d_nov, d_ct);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  prep<<<(*nov + threads - 1) / threads, threads>>>(d_xadj, d_adj, d_nov, k, k,
                                                    d_ct);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaMemcpy(ct, d_ct, (*nov) * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < *nov; i++)
    printf("%d %d\n", i, ct[i]);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU scale took: %f s\n", elapsedTime / 1000);
  cudaFree(d_xadj);
  cudaFree(d_adj);
  cudaFree(d_nov);
  cudaFree(d_ct);
}

/*Read the given file and return CSR*/
void *read_edges(char *bin_name, int k) {
  std::cout << "fname: " << bin_name << std::endl;

  // count the newlines
  unsigned int number_of_lines = 0;
  FILE *infile = fopen(bin_name, "r");
  int ch;
  while (EOF != (ch = getc(infile)))
    if ('\n' == ch)
      ++number_of_lines;
  ++number_of_lines;
#ifdef DEBUG
  std::cout << number_of_lines << " lines" << std::endl;
#endif
  fclose(infile);

  // read the first line, set it to no vertices.
  std::ifstream bp(bin_name);
  int *no_vertices = new int;
  std::string line;

  int i, j, max = 0;
  for (int iter = 0; iter < number_of_lines; iter++) {
    std::getline(bp, line);
    std::istringstream myss(line);
    if (!(myss >> i >> j)) {
      break;
    }
    if (i > max)
      max = i;
    if (j > max)
      max = j;
  }
  bp.clear();
  bp.seekg(0);
  *no_vertices = max + 1;
  int no_edges = (number_of_lines)*2; // bidirectional
  /*TODO unique and no loop decreases this, we should resize adj accordingly.
   * Not the end of the world, we will never reach those indices.*/

  // if file ended with \n you'd keep it as is.
  // std::cout << "allocating A: " << sizeof(std::vector<int>) * *no_vertices
  //          << "bytes. " << *no_vertices << " vectors." << std::endl;

  std::vector<int> *A = new std::vector<int>[*no_vertices];
  // std::cout << "allocated A" << std::endl;

  for (int iter = 0; iter < number_of_lines; iter++) {
    std::getline(bp, line);
    std::istringstream myss(line);
    if (!(myss >> i >> j)) {
      break;
    }

#ifdef DEBUG
    std::cout << i << " " << j << std::endl;
#endif
    if (i != j) {
      // ignore diagonal edges
      A[i].push_back(j);
      A[j].push_back(i);
    }
  }
  for (int i = 0; i < *no_vertices; i++) {
    std::sort(A[i].begin(), A[i].end());
    // sort then unique.
    // you may have 3 1 and 1 3
    // if you do not sort, unique doesn't do what I think it would.
    // also we prefer them sorted in case the file has 1 2 before 1 0 or sth.
    // using default comparison:
    std::vector<int>::iterator it;
    it = std::unique(A[i].begin(), A[i].end());   // 10 20 30 20 10 ?  ?  ?  ?
                                                  //                ^
    A[i].resize(std::distance(A[i].begin(), it)); // 10 20 30 20 10
  }
  int sum = 0;
  int *xadj = new int[*no_vertices + 1]; // last one marks the end of the adj.
  int *adj = new int[no_edges]; // there are m+1 lines (m '\n's), 2m edges.
  xadj[0] = 0;
  for (int i = 0; i < *no_vertices; i++) {
    // adj.add_to_end(A[i])
    for (int j = 0; j < A[i].size(); j++) {
      adj[sum + j] = A[i][j];
    }
    sum += A[i].size();
    xadj[i + 1] = sum;
  }
  std::cout << "Done reading." << std::endl;
  wrapper(xadj, adj, no_vertices, no_edges, k);
  return 0;
}

int main(int argc, char *argv[]) {
  /*first arg is filename, second is k*/
  // omp_set_num_threads(8);
  read_edges(argv[1], atoi(argv[2]));
  return 0;
}
