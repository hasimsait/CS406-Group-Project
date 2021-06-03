# CS406-Group-Project

Haşim Sait Göktan

Kerem Güneş

Kemal Şeker

Onur Arda Bodur

## Usage:
    module load gcc/7.5.0
    module load cuda/10.0
    nvcc kernel.cu -O3 -Xcompiler -O3 -o gpu
    module unload cuda/10.0
    module unload gcc/7.5.0
    module load gcc/8.2.0 #the latest version you have installed
    g++ main.cpp -O3 -fopenmp -o cpu
    export OMP_NUM_THREADS=thread_count;
	#(decomment lines 107 and 150 to print output, only the timing will be printed otherwise)
    ./gpu filename k
    ./cpu filename k
    
    
## Results on provided graphs
| instance    | k | sequential | OpenMP t=2 | OpenMP t=4 | OpenMP t=8 | OpenMP t=16 | Cuda        |
| ----------- | - | ---------- | ---------- | ---------- | ---------- | ----------- | ----------- |
| amazon      | 3 |  0.279555  |  0.194179  |  0.133981  |  0.0905637 |  0.0699001  |  0.071476   |
| amazon      | 4 |  1.64887   |  0.915767  |  0.513519  |  0.301799  |  0.173176   |  1.167217   |
| amazon      | 5 |  10.5503   |  5.97017   |  3.02007   |  1.58916   |  0.838295   |  15.079791  |
| dblp        | 3 |  0.495958  |  0.356659  |  0.257321  |  0.165979  |  0.109911   |  0.679071   |
| dblp        | 4 |  12.0967   |  8.18795   |  6.58545   |  5.35292   |  4.64312    |  50.470917  |        
| dblp        | 5 |  629.311   |  514.97    |  584.028   |  546.786   |  482.425    |  >1200      |

## Cuda results on provided graphs after passing one path per thread only
| instance    | k |  local/global path |  shared path and counter |no recursion| >start (atomic) | multiple counters then reduce |
| ----------- | - | ------------------ | ------------------------ | ---------- | --------------- | ----------------------------- |
| amazon      | 3 |  0.279176          |  0.216811                |  0.098119  |  0.071492       |  0.073181                     |
| amazon      | 4 |  3.601732          |  3.115706                |  1.340244  |  1.143491       |  1.163768                     |
| amazon      | 5 |  54.977879         |  48.157932               |  18.542461 |  12.318878      |  12.341285                    |
| dblp        | 3 |  1.975520          |  1.929673                |  0.942191  |  0.684884       |  0.699602                     |
| dblp        | 4 |  188.793884        |  188.082703              |  84.935524 |  138.882248     |  257.226624                   |
| dblp        | 5 |  >1200             |  >1200                   |  >1200     |  >1200          |  >1200                        |

The local and global memory accesses are around the same as excepted therefore the differences were within margin of error.<br>
Increasing stride of shared path or counter increases the time. I tought the malloc would have a larger impact.<br>
Adding the >start reduces the search space by k, but the cost of atomic operations is too large.<br> In order to improve the atomicAdd's turnaround time, we split the count array into k seperate global arrays (doesn't fit in shared), then reduce. While this approach improves the times on amazon, it slows larger k on dblp.<br>
We roll back to the no recursion (unfold, turn each k into global functions) approach for k>3. 3 atomics per iteration is balanced by the reduced number of iterations.<br>
All marked array does in k=3 can be accomplished by adj[i]>start, adj[j]>adj[i] and has been added with >start to the k=3, which is the main reason behind the speedup. On larger k, marked array is necessary since loops like 1,4,3,2 can not be found with next>current vertex.

*stride=1, once you take the transpose of the adj and a, it will speed up significantly. This DFS approach is limited on gpus as it causes significant divergence. When you have a certain amount of iterations of the for loop that performs the recursive call instead of xadj[i+1]-xadj[i], the performance will improve. The representation that provides stride also reduces the divergence significantly.

The marked array is too large to create N copies of, initializing the pointers to those arrays crashes due to invalid memory access. This causes our parallel implementation to execute more instructions for all k and n than the sequential implementation.

The greatest limit seems to be using more memory than what is available on the device. A way to reduce the memory usage is to use linked lists instead of paths of integers to store and pass the paths.

Moving the count array to the shared memory would also for obvious reasons. Think of the results above as pre optimization, all steps are known and we have a working, correct implementation.
