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
    g++ counter_implemented.cpp -O3 -fopenmp -o cpu
	export OMP_NUM_THREADS=thread_count;
	(decomment lines 267 and 311 to print output, just timing will be printed otherwise)
    ./gpu filename k
    ./cpu filename k
    
    
## Results on provided graphs
| instance    | k | sequential | OpenMP t=2 | OpenMP t=4 | OpenMP t=8 | OpenMP t=16 | Cuda        |
| ----------- | - | ---------- | ---------- | ---------- | ---------- | ----------- | ----------- |
| amazon      | 3 |  0.279555  |  0.194179  |  0.133981  |  0.0905637 |  0.0699001  |  1.09738    |
| amazon      | 4 |  1.64887   |  0.915767  |  0.513519  |  0.301799  |  0.173176   |  6.85345    |
| amazon      | 5 |  10.5503   |  5.97017   |  3.02007   |  1.58916   |  0.838295   |  57.700562  |
| dblp        | 3 |  0.495958  |  0.356659  |  0.257321  |  0.165979  |  0.109911   |  2.025606   |
| dblp        | 4 |  12.0967   |  8.18795   |  6.58545   |  5.35292   |  4.64312    |  191.320938 |        
| dblp        | 5 |  629.311   |  514.97    |  584.028   |  546.786   |  482.425    |  >600       |


NOTE: The timings are made after the fixes we mentioned that we just did before demo. This is still not fully optimized, especially CUDA. It will be worked on starting tonight.
*stride=1, once you take the transpose of the adj and a, it will speed up significantly. This DFS approach is limited on gpus as it causes significant divergence. When you have a certain amount of iterations of the for loop that performs the recursive call instead of xadj[i+1]-xadj[i], the performance will improve. The representation that provides stride also reduces the divergence significantly.

The marked array is too large to create N copies of, initializing the pointers to those arrays crashes due to invalid memory access. This causes our parallel implementation to execute more instructions for all k and n than the sequential implementation.

The greatest limit seems to be using more memory than what is available on the device. A way to reduce the memory usage is to use linked lists instead of paths of integers to store and pass the paths.

Moving the count array to the shared memory would also for obvious reasons. Think of the results above as pre optimization, all steps are known and we have a working, correct implementation.
