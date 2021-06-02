#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#define THREADS 64
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

__global__ void deviceDFSk5(int *xadj, int *adj, int *nov, int *counter) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int ct[THREADS];
  if (id < *nov) {
    ct[threadIdx.x] = 0;
    for (int i = xadj[id]; i < xadj[id + 1]; i++) {
      // adj[i] are the neighbors of id vertex, none can be id by
      // definition (no loops of len-1)
      if (adj[i] != id) {
        for (int j = xadj[adj[i]]; j < xadj[adj[i] + 1]; j++) {
          // adj[j] are the neighbors of the second vertex on this path,
          // they can't be id.
          if (adj[j] != adj[i] && adj[j] != id) {
            for (int k = xadj[adj[j]]; k < xadj[adj[j] + 1]; k++) {
              // adj[k] are the neighbors of the third vertex,
              // they can't be equal to id or the second.
              if (adj[k] != adj[j] && adj[k] != adj[i] && adj[k] != id) {
                for (int l = xadj[adj[k]]; l < xadj[adj[k] + 1]; l++) {
                  // adj[l] are the neighbors of the fourth vertex,
                  // they can't be equal to id second or the third.
                  if (adj[l] != adj[k] && adj[l] != adj[j] &&
                      adj[l] != adj[i] && adj[l] != id) {
                    for (int m = xadj[adj[l]]; m < xadj[adj[l] + 1]; m++) {
                      // adj[l] are the neighbors of the fourth vertex,
                      // they can't be equal to id second or the third.
                      if (adj[m] == id) {
                        ct[threadIdx.x]++;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    counter[id] = ct[threadIdx.x];
  }
}

__global__ void deviceDFSk4(int *xadj, int *adj, int *nov, int *counter) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int ct[THREADS];
  if (id < *nov) {
    ct[threadIdx.x] = 0;
    for (int i = xadj[id]; i < xadj[id + 1]; i++) {
      // adj[i] are the neighbors of id vertex, none can be id by
      // definition (no loops of len-1)
      if (adj[i] != id) {
        for (int j = xadj[adj[i]]; j < xadj[adj[i] + 1]; j++) {
          // adj[j] are the neighbors of the second vertex on this path,
          // they can't be id.
          if (adj[j] != adj[i] && adj[j] != id) {
            for (int k = xadj[adj[j]]; k < xadj[adj[j] + 1]; k++) {
              // adj[k] are the neighbors of the third vertex,
              // they can't be equal to id or the second.
              if (adj[k] != adj[j] && adj[k] != adj[i] && adj[k] != id) {
                for (int l = xadj[adj[k]]; l < xadj[adj[k] + 1]; l++) {
                  // adj[l] are the neighbors of the fourth vertex,
                  if (adj[l] == id) {
                    ct[threadIdx.x]++;
                  }
                }
              }
            }
          }
        }
      }
    }
    counter[id] = ct[threadIdx.x];
  }
}

__global__ void deviceDFSk3(int *xadj, int *adj, int *nov, int *counter) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < *nov) {

    for (int i = xadj[id]; i < xadj[id + 1]; i++) {
      if (adj[i] > id) {
        for (int j = xadj[adj[i]]; j < xadj[adj[i] + 1]; j++) {
          if (adj[j] > adj[i] && adj[j] > id) {
            for (int k = xadj[adj[j]]; k < xadj[adj[j] + 1]; k++) {
              if (adj[k] == id) {
                atomicAdd(&counter[id], 2);
                atomicAdd(&counter[adj[i]], 2);
                atomicAdd(&counter[adj[j]], 2);
              }
            }
          }
        }
      }
    }
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
  if (k == 3) {
    /*not that necessary, to ensure d_ct is set to zero in the least amount of
     * lines possible for k=3*/
    memset(ct, 0, (*nov) * sizeof(int));
    cudaMemcpy(d_ct, ct, (*nov) * sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_xadj, xadj, (*nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_adj, adj, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nov, nov, sizeof(int), cudaMemcpyHostToDevice);

  gpuErrchk(cudaDeviceSynchronize());
#ifdef DEBUG
  std::cout << "malloc copy done" << std::endl;
#endif
  gpuErrchk(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  if (k == 3)
    deviceDFSk3<<<(*nov + THREADS - 1) / THREADS, THREADS>>>(d_xadj, d_adj,
                                                             d_nov, d_ct);
  if (k == 4)
    deviceDFSk4<<<(*nov + THREADS - 1) / THREADS, THREADS>>>(d_xadj, d_adj,
                                                             d_nov, d_ct);
  if (k == 5)
    deviceDFSk5<<<(*nov + THREADS - 1) / THREADS, THREADS>>>(d_xadj, d_adj,
                                                             d_nov, d_ct);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaMemcpy(ct, d_ct, (*nov) * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < *nov; i++)
    printf("%d %d\n", i, ct[i]);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU took: %f s\n", elapsedTime / 1000);
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
  /*TODO unique and no loop decreases this, we should resize adj
   * accordingly. Not the end of the world, we will never reach those
   * indices.*/

  // if file ended with \n you'd keep it as is.

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
    // also we prefer them sorted in case the file has 1 2 before 1 0 or
    // sth. using default comparison:
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