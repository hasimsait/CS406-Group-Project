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
    ./gpu filename k
    ./cpu filename k
    
    
## Results on provided graphs
| instance    | k | sequential |  OpenMP t=2 | OpenMP t=4 | OpenMP t=8 | OpenMP t=16 | Cuda        |
| ----------- | - | -----------|  --------   |   -------- |   -------- |   --------- | ----------- |
| amazon      | 3 |  1.12096   |  0.271340   |    1.1042  |    1.11375 |    1.11984  |  1.09738    |
| amazon      | 4 |  6.89942   |  3.790840   |    7.25519 |    6.95469 |    6.89237  |  6.85345    |
| amazon      | 5 |  53.2337   |  53.1851    |    53.0025 |    52.9501 |    52.9904  |  57.700562  |
| dblp        | 3 |  1.69241   |  1.69536    |    1.69693 |    1.69241 |    1.64419  |  2.025606   |
| dblp        | 4 |  39.3576   |  43.3626    |    39.2935 |    39.4813 |    39.2992  |  191.320938 |        
| dblp        | 5 |  >600      |  >600       |    >600    |    >600    |    >600     |  >600       |


*stride=1, once you take the transpose of the adj and a, it will speed up significantly. This DFS approach is limited on gpus as it causes significant divergence. When you have a certain amount of iterations of the for loop that performs the recursive call instead of xadj[i+1]-xadj[i], the performance will improve. The representation that provides stride also reduces the divergence significantly.

The marked array is too large to create N copies of, initializing the pointers to those arrays crashes due to invalid memory access. This causes our parallel implementation to execute more instructions for all k and n than the sequential implementation.

The greatest limit seems to be using more memory than what is available on the device. A way to reduce the memory usage is to use linked lists instead of paths of integers to store and pass the paths.

Moving the count array to the shared memory would also for obvious reasons. Think of the results above as pre optimization, all steps are known and we have a working, correct implementation.
