#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h> /* fabs */
#include <memory>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

/* TARGET, quoting from Kamer hoca's mail:
"In the course project, you will be given an undirected graph G and a number k.
A graph G = (V, E) has its set of vertices in V and its set of edges in E. The
edges are undirected. That is if {u, v} is in E then u is connected to v, and v
is also connected to u. For a given G and a positive integer 2 < k < 6, your
program must find the number of length k cycles containing the vertex u for each
u in V. For instance, u-v-x-u is a length-3 cycle which will contribute to the
final value of u, v, and x if k is 3.

You will implement a sequential version, an OpenMP-based multicore CPU version,
and a CUDA-based GPU version and show the speedups with explanations. In
addition, if you can get speedup over the GPU-based version by using both CPU
and GPU you will receive bonus points.

Each graph file will contain m+1 lines where m is the number of edges in the
graph. Each file will contain the following information.

    no_vertices
    u1 v1
    u2 v2
    ....
    um jm

In the file, for each line ui < vi. You need to add the other orientation to the
CSR data structure. The program will be executed as

./executable path_to_file k"
*/

/*proposed DFS:
  function DFS(graph, marked, k, vertex, start, count)
    marked[vertex] = True
    if(k == 0)
      marked[vertex] = False
      if(vertex and start are adjacent)
        count +=1
      return count
    for all v in V adjacent to vertex
      if(!marked[v])
        count = DFS(graph, marked, k-1, v, start, count)
    marked[vertex] = False
    return count

  let  marked be  boolean array of size V
  for k <- 3 to 5
    for v <- 0 to V
      marked[v] = False
    let count = 0
    for v <- 0 to V-(k-1)
      count = DFS(graph,marked,k-1,v,v,count)
      marked[v] = True
*/
void parallel_k_cycles(int *xadj, int *adj, int *nov, int k) {
  /*ADJ AND XADJ ARE CORRECT, UNCOMMENT TO VERIFY.*/
  /*
  for (int i = 0; i < *nov + 1; i++) {
    std::cout << xadj[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < *nov; i++) {
    for (int j = xadj[i]; j < xadj[i + 1]; j++) {
      std::cout << adj[j] << " ";
    }
  }
  */
  // TODO OPEN_MP IMPLEMENTATION OF THE DFS THAT WAS PROPOSED
}

/*Read the given file and return CSR*/
void *read_edges(std::string bin_name, int k) {
  std::cout << "fname: " << bin_name << std::endl;

  // count the newlines
  unsigned int number_of_lines = 0;
  FILE *infile = fopen(bin_name.c_str(), "r");
  int ch;
  while (EOF != (ch = getc(infile)))
    if ('\n' == ch)
      ++number_of_lines;
  ++number_of_lines;
  // std::cout << number_of_lines << " lines" << std::endl;
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
    // ignore diagonal edges
    // you may also have 3 1 and 1 3
    // std::cout << i << " " << j << std::endl;
    if (i != j) {
      A[i].push_back(j);
      A[j].push_back(i);
    }
  }
  for (int i = 0; i < *no_vertices; i++) {
    std::sort(A[i].begin(), A[i].end());
    // sort then unique.
    std::vector<int>::iterator ip;
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
  int end_of_adj = 0;
  for (int i = 0; i < *no_vertices; i++) {
    // adj.add_to_end(A[i])
    for (int j = 0; j < A[i].size(); j++) {
      adj[sum + j] = A[i][j];
    }
    sum += A[i].size();
    xadj[i + 1] = sum;
  }
  std::cout << "Done reading." << std::endl;
  parallel_k_cycles(xadj, adj, no_vertices, k);
  return nullptr;
}

int main(int argc, char *argv[]) {
  /*first arg is filename, second is k*/
  read_edges(argv[1], atoi(argv[2]));
  return 0;
}