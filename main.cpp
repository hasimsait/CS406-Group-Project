#include <algorithm>
#include <deque>
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

struct queue_element {
  std::vector<int> prev_vertices;
  int k;
  int curr_vertex;
  int start;
};

// if vertex's neighbors include start
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

Each graph file will contain m lines where m is the number of edges in the
graph. Each file will contain the following information.

    u1 v1
    u2 v2
    ....
    um jm

You need to add the other orientation to the CSR data structure. The program
will be executed as

./executable path_to_file k"
*/

/*returns true if array contains item within [start,end)*/
bool contains(int *array, int start, int end, int item) {
  /*TODO array[start] to array[end] is sorted, switch to binary search*/
  for (int j = start; j < end; j++) {
    if (array[j] == item)
      return true;
  }
  return false;
}

void BFS(int *xadj, int *adj, int *nov, std::vector<int> prev_vertices, int k,
         int curr_vertex, int start, int &ct, std::deque<queue_element> &work) {
  // this has a memory issue.
  if (k == 0) {
    if (contains(adj, xadj[curr_vertex], xadj[curr_vertex + 1], start)) {
      // if vertex's neighbors include start
      ct++;
    }
  }
  for (int j = xadj[curr_vertex]; j < xadj[curr_vertex + 1]; j++)
    if (std::find(prev_vertices.begin(), prev_vertices.end(), curr_vertex) ==
        prev_vertices.end())
    // prev vertices do not include curr_vertex)
    {
      struct queue_element newelem;
      std::vector<int> cp = prev_vertices;
      cp.push_back(curr_vertex);
      newelem.prev_vertices = cp;
      newelem.k = k - 1;
      newelem.curr_vertex = j;
      newelem.start = start;
      work.push_back(newelem);
    }
}
void BFS_driver(int *xadj, int *adj, int *nov, int k) {
  std::deque<queue_element> work;
  int count = 0;

  for (int i = 0; i < *nov; i++) {
    struct queue_element newelem;
    std::vector<int> cp;
    newelem.prev_vertices = cp;
    newelem.k = k - 1;
    newelem.curr_vertex = i;
    newelem.start = i;
    work.push_back(newelem);
  }
  while (!work.empty()) {
    BFS(xadj, adj, nov, work.front().prev_vertices, work.front().k,
        work.front().curr_vertex, work.front().start, count, work);
    work.pop_front();
  }
  std::cout << count << std::endl;
}

void DFS(int *xadj, int *adj, int *nov, bool *marked, int k, int vertex,
         int start, int &count) {
  /*this should've worked*/
  /*
  marked[vertex] = true;
  if (k == 0) {
    marked[vertex] = false;
    if (contains(adj, xadj[vertex], xadj[vertex + 1], start)) {
      // if vertex's neighbors include start
      count++;
    }
    return count;
  }
  for (int j = xadj[vertex]; j < xadj[vertex + 1]; j++) {
    if (!marked[adj[j]])
      count = DFS(xadj, adj, nov, marked, k - 1, vertex, start, count);
  }
  marked[vertex] = false;
  return count;
  */
  // mark the vertex vert as visited
  marked[vertex] = true;

  // if the path of length (n-1) is found
  if (k == 0) {

    // mark vert as un-visited to make
    // it usable again.
    marked[vertex] = false;

    // Check if vertex vert can end with
    // vertex start
    if (contains(adj, xadj[vertex], xadj[vertex + 1], start)) {
      (count)++;
      return;
    } else
      return;
  }

  // For searching every possible path of
  // length (n-1)
  for (int j = xadj[vertex]; j < xadj[vertex + 1]; j++)
    if (!marked[adj[j]])
      // DFS for searching path by decreasing length by 1
      DFS(xadj, adj, nov, marked, k - 1, vertex, start, count);

  // marking vert as unvisited to make it
  // usable again.
  marked[vertex] = false;
}

int parallel_k_cycles(int *xadj, int *adj, int *nov, int k) {
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
  return 0;
}
int sequential_k_cycles(int *xadj, int *adj, int *nov, int k) {
  /* this should've worked*/
  /*
  bool *marked = new bool[*nov];
  for (int i = 0; i < *nov; i++) {
    marked[i] = false;
  }
  int count = 0;
  for (int i = 0; i < *nov - (k - 1); i++) {
    count = DFS(xadj, adj, nov, marked, k - 1, i, i, count);
    marked[i] = true;
  }
  // the contribute line mentons this
  std::cout << (count / 2) * k << " loops" << std::endl;
  return (count / 2) * k;*/

  // all vertex are marked un-visited initially.
  bool *marked = new bool[*nov];

  for (int i = 0; i < *nov; i++) {
    marked[i] = false;
  }
  // Searching for cycle by using v-n+1 vertices
  int ct = 0;
  // int *count = &ct;
  for (int i = 0; i < *nov - (k - 1); i++) {
    DFS(xadj, adj, nov, marked, k - 1, i, i, ct);
    // DFS(graph,marked,3,0,0,0)->DFS(graph,marked,3,1,1,value of
    // ct)->DFS(graph,marked,3,2,2,value of ct)

    // ith vertex is marked as visited and
    // will not be visited again.
    marked[i] = true;
  }
  /*the contributes line mentions this*/
  std::cout << (ct / 2) * k << std::endl;
  return (ct / 2) * k;
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
  // sequential_k_cycles(xadj, adj, no_vertices, k);
  // BFS_driver(xadj, adj, no_vertices, k);
  return nullptr;
}

int main(int argc, char *argv[]) {
  /*first arg is filename, second is k*/
  read_edges(argv[1], atoi(argv[2]));
  return 0;
}