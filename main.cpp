#include <algorithm>
#include <cstring>
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

struct work {
  // previous vertices in this path. k max is 5 anyways.
  std::vector<int> prev_vertices;
  // number of vertices that will be added to this path
  int k;
  // the vertex that we're adding to this path in this branch
  int curr_vertex;
};

/*returns true if array contains item within [start,end)*/
bool contains(int *array, int start, int end, int item) {
  /*TODO array[start] to array[end] is sorted, switch to binary search*/
  for (int j = start; j < end; j++) {
    if (array[j] == item)
      return true;
  }
  return false;
}

int find_in_path(int *array, int item, int length) {
  int i = 0;
  while (array[i] != -1 && i < length) {
    if (i == item)
      return i;
    i++;
  }
  return -1;
}

void parallel_BFS(int *xadj, int *adj, int *nov, int *prev_vertices, int k,
                  int curr_vertex, int curr_len, int &ct) {
#ifdef DEBUG
  std::cout << "running bfs on " << curr.k << " " << curr.curr_vertex
            << std::endl;
#endif
  if (k == 0) {
    if (contains(adj, xadj[curr_vertex], xadj[curr_vertex + 1],
                 prev_vertices[0])) {
      // if vertex's neighbors include start
#pragma omp atomic
      ct++;
// avoid race condition
#ifdef DEBUG
      std::cout << "added one loop!" << std::endl;
      for (int i = 0; i < 5; ++i)
        std::cout << prev_vertices[i] << ' ';
      std::cout << curr_vertex << std::endl;
#endif
    }
  }
#pragma omp parallel
#pragma omp single
  {
    /*no need for path to be private, all neighbors share the same path*/
    int *path = new int[curr_len + 1];
    memcpy(path, prev_vertices, sizeof(*prev_vertices));
    // last element will be undefined, rest will be copied from
    // prev_vertices.
    path[curr_len - 1] = curr_vertex;
    path[curr_len] = -1;
    // TODO we're passing count along, no need for this poor man's \0
    for (int j = xadj[curr_vertex]; j < xadj[curr_vertex + 1]; j++)
      if (find_in_path(prev_vertices, adj[j], curr_len) == -1 && k != 0)
      // prev vertices do not include the neighbor we're attempting to insert)
      {
#pragma omp task
        { parallel_BFS(xadj, adj, nov, path, k - 1, adj[j], curr_len + 1, ct); }
      }
  }
}

void parallel_BFS_driver(int *xadj, int *adj, int *nov, int k) {
  std::cout << "Parallel BFS with OMP is starting" << std::endl;
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
  int count = 0;
  double start = omp_get_wtime();
#pragma omp parallel
#pragma omp single
  {
    int *prev = new int[1];
    prev[0] = -1;
    for (int i = 0; i < *nov; i++) {
#pragma omp task
      { parallel_BFS(xadj, adj, nov, prev, k - 1, i, 1, count); }
    }
  }
#pragma omp taskwait
  double end = omp_get_wtime();
  std::cout << "Total cycles of length " << k << " are " << count / 2
            << " it took " << end - start << " seconds." << std::endl;
}

void BFS(int *xadj, int *adj, int *nov, work curr, int &ct,
         std::deque<work> &work_queue) {
#ifdef DEBUG
  std::cout << "running bfs on " << curr.k << " " << curr.curr_vertex
            << std::endl;
#endif
  if (curr.k == 0) {
    if (contains(adj, xadj[curr.curr_vertex], xadj[curr.curr_vertex + 1],
                 curr.prev_vertices[0])) {
      // if vertex's neighbors include start
      ct++;
#ifdef DEBUG
      std::cout << "added one loop!" << std::endl;
      for (int i = 0; i < curr.prev_vertices.size(); ++i)
        std::cout << curr.prev_vertices[i] << ' ';
      std::cout << curr.curr_vertex << std::endl;
#endif
    }
  }
  for (int j = xadj[curr.curr_vertex]; j < xadj[curr.curr_vertex + 1]; j++)
    if (std::find(curr.prev_vertices.begin(), curr.prev_vertices.end(),
                  adj[j]) == curr.prev_vertices.end() &&
        curr.k != 0)
    // prev vertices do not include the neighbor we're attempting to insert)
    {
      struct work new_work;
      new_work.prev_vertices = curr.prev_vertices;
      new_work.prev_vertices.push_back(curr.curr_vertex);
      new_work.k = curr.k - 1;
      new_work.curr_vertex = adj[j];
      work_queue.push_back(new_work);
    }
}

void BFS_driver(int *xadj, int *adj, int *nov, int k) {
  std::cout << "Sequential BFS is starting" << std::endl;
  std::deque<work> work_queue;
  int count = 0;
  double start = omp_get_wtime();
  for (int i = 0; i < *nov; i++) {
    struct work new_work;
    std::vector<int> cp;
    new_work.prev_vertices = cp;
    new_work.k = k - 1;
    new_work.curr_vertex = i;
    work_queue.push_back(new_work);
  }
  while (!work_queue.empty()) {
    BFS(xadj, adj, nov, work_queue.front(), count, work_queue);
    work_queue.pop_front();
  }
  double end = omp_get_wtime();
  std::cout << "Total cycles of length " << k << " are " << count / 2
            << " it took " << end - start << " seconds." << std::endl;
}

void DFS(int *xadj, int *adj, int *nov, bool *marked, int k, int vertex,
         int start, int &count) {
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
      // std::cout << "count incremented";
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
      DFS(xadj, adj, nov, marked, k - 1, adj[j], start, count);

  // marking vert as unvisited to make it
  // usable again.
  marked[vertex] = false;
}

int sequential_k_cycles(int *xadj, int *adj, int *nov, int k) {
  std::cout << "Sequential DFS is starting" << std::endl;
  // all vertex are marked un-visited initially.
  bool *marked = new bool[*nov];

  for (int i = 0; i < *nov; i++) {
    marked[i] = false;
  }
  // Searching for cycle by using v-n+1 vertices
  int ct = 0;
  // int *count = &ct;
  double start = omp_get_wtime();
  for (int i = 0; i < *nov - (k - 1); i++) {
    DFS(xadj, adj, nov, marked, k - 1, i, i, ct);
    // DFS(graph,marked,3,0,0,0)->DFS(graph,marked,3,1,1,value of
    // ct)->DFS(graph,marked,3,2,2,value of ct)

    // ith vertex is marked as visited and
    // will not be visited again.
    marked[i] = true;
  }
  /*the contributes line mentions this*/
  double end = omp_get_wtime();
  std::cout << "Total cycles of length " << k << " are " << ct / 2 * k
            << " it took " << end - start << " seconds." << std::endl;
  return (ct / 2) * k;
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
  sequential_k_cycles(xadj, adj, no_vertices, k);
  BFS_driver(xadj, adj, no_vertices, k);
  parallel_BFS_driver(xadj, adj, no_vertices, k);
  return 0;
}

int main(int argc, char *argv[]) {
  /*first arg is filename, second is k*/
  omp_set_num_threads(8);
  read_edges(argv[1], atoi(argv[2]));
  return 0;
}