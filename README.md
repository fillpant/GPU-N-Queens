# The N-Queens Problem
The N-Queens problem asks how many possible placements of queens exist on an $N\times N$ chess board such that no queen can attack an other. A queen can attack another if it is in the same row, column or diagonals as it. In other words, this is about counting **all** valid solutions, not finding a subset of them! 

An enormous effort is required to tackle larger N with solutions only known for N=27 as of yet *(Aug 2022)* which remains unverified. Purpose-built FPGAs have been used to tackle N=26 and N=27, however the wide availability of programmable GPUs (GPGPUs) may allow for N=28 to be found in a cluster of GPUs! 

This repo contains the source code for an optimised CUDA solver currently implementing `DoubleSweep-light`, which is a version of the `DoubleSweep` algorithm presented in the OKlibrary<sup>[1]</sup>. 

This is very much work in progress, and is not intended for deployment in its current state as major changes are planned (see *Future Work*), however all feedback is always welcome!

## High Level Overview of Algorithm
Solving the N-Queens problem must be done using efficient algorithms to stand a chance of finding all solutions in a reasonable time frame. 

A (very) naïve approach to solving the problem is placing $N$ queens on an  $N\times N$ board in the same row or column, and indiscriminately move them (binary addition style), checking each time if a valid solution has formed. This will explore $N*N\choose N$ placements, which is a huge number even for smaller $N$. For example, for $N=15$, ${15*15\choose 15} = 91,005,567,811,177,478,095,440$ placements will be explored!

A (better) approach is to use backtracking to place queens from one side of the board to another, and enforcing some simple rules in the process:
 - No queen can be placed in the same row as another
 - No queen can be placed in the same column as another
 - No queen can be on the same diagonal (or anti-diagonal) as another

Already, the number of arrangements that must be tried is smaller, but a lot of 'fruitless paths' will be tried resulting in a 'dead end' before all queens are placed on the board. 

The `DoubleSweep-light` algorithm uses the backtracking approach above, and a subset of the features of `DoubleSweep`, aimed at preventing some of the 'fruitless paths' from being explored. This is based on the fact that when a queen is placed on a row of the board, subsequent rows may become 'unit' (i.e. for them to have a queen placed on them, there is exactly one position where the queen won't conflict with another). Hence, upon placement of a queen, 'propagation' is started to find which rows that follow are unit, and populate them.

## This Solver
This repo houses the code for an attempt at implementing the above algorithm efficiently. Designed to run on a wide range of microarchitectures, and be easily scalable and extensible. 

The hope is that one day this solver will be at a state where it can be used on a large cluster to solve a 'large' value of $N$. 

Optimisation effort has been put specifically on the instruction level, and design. More optimisations on the algorithmic level are underway, detailed in the future work section

## How it Works
The basic idea here is that we want a large number of parallel solvers running on the GPU, potentially across many GPUs (not necessarily in the same machine). To achieve this, a large pool of distinct 'initial states' is generated, each being an N-Queens board with some queens placed on it in a valid configuration.

Then, each thread on the GPU is given one of the states from the pool to explore. Threads only communicate between them in their respective warps, and this is done primarily to coordinate thread execution and minimise divergence, otherwise the model requires no communication between threads at all! Threads are communicating between them in their warps to decide when they are all ready to stop. This is done by balloting the threads on each step of the computation with the question 'Are you done counting', and only once all threads have voted 'Yes', summarisation of results can begin.

Once all threads in a warp are done counting solutions for their respective states, a parallel reduction is performed to sum up the warp's solution count. When all warps in a block finish, then their sums of solutions are put through a final parallel reduction to be summarised to the block's sum of solution. This is written (in distinct locations) in global memory from where the host(s) can retrieve. 

## Future Work
 The following (unordered, non-exhaustive) items is our future work, with some items currently under way.
 ### Fully implemented the `DoubleSweep` algorithm
 The full algorithm consists of more clever techniques to further reduce the work that needs to be carried out but poses some challenges to completely implement on the GPU and maintain our scalability goal, but is possible!
  
 ### Implement transformation elimination
 It is often the case that some search paths converge into the same fundamental solution with a transformation applied to it (90/180/270 degree rotation or horizontal/vertical mirroring). Currently, multiple (different) threads carry out work and come to the same solution(s) under different orientations or transformations. These paths can be avoided all together!
 
 ### Devise a restart heuristic for longer running computations
With irregular computations like this, it is likely the case that some threads finish the work they are assigned sooner than others. This results in warps having fewer active threads at any moment, and thus reduces their throughput since only a subset of their threads are actually using up resources and the rest are sitting idle. For smaller computations this doesn't have a significant impact, however longer computations may have a more pronounced difference in thread execution times. The current mitigation to the problem is to have a very big pool of initial states so that threads perform more, shorter computations.  
 
A restart may help reduce warp 'idleness', since the new kernel would re-group active threads into new warps. The question now is, how do we decide when the right time for a restart is? We need some heuristic(s) for this! The following are to be explored:
 - **Warp activity based**
	 - Adjust thread balloting so threads decide to exit when some portion of the warp is inactive. Alive threads write their current state to global memory, and wait for all warps to exit. When the kernel exits, the host notices the computation is incomplete, and re-starts it over the remaining states.
- **Randomised / Scheduled**
	- Restarts happen randomly, potentially influenced by some information provided from all warps (such as the current rate of active/idle threads in warps)
	- Restarts are scheduled to occur after X seconds/clock ticks/other metric.
- **Host triggered**
	- The host observes some indirect progress indicator (GPU load etc) and chooses to interrupt the computation and restart (likely going to be problematic but worth a try)

## More Techy Bits
The header `deffinitions.cuh` contains some macro definitions acting as configuration parameters for the solver detailed bellow:

`N` The value of $N$ to solve. Consequently, to solve different N the code has to be recompiled with an updated value to this definition. **Warning: this definition may conflict with some windows headers** 

`NQ_ENABLE_EXPERIMENTAL_OPTIMISATIONS` if defined, will enable certain optimisations that may or may not result in better performance. Usually they come in the form of inline assembly and yield performance improvements on some devices (GTX 1080ti and RTX 3090 under CUDA 11.7 tested). These optimisations may result in worse performance in some cases so this must be tested before use.

`MINIMUM_COMPUTE_CAPABILITY` is an integer definition with the minimum compute capability required to run this tool. An integer 600 means compute capability 6.0 for instance.

`PROFILING_ROUNDS` Used during kernel profiling. If defined, changes the behaviour of the solver driver to run this many rounds of profiling, consequently changing its output too. During profiling results for the $N$ solved may be incorrect and are hence ignored. 

`N_MASK` A bit expression evaluating to a mask with the least significant $N$ bits set. 

`STATE_GENERATION_DOWNSLOPE_BOUNDED` If set to `1` will constraint the initial state pool generation to stop when a downslope in state generation is detected. More specifically, if $X$ states are requested from state generation, it will attempt to lock at some row $R$. If the number of generated states is bellow $X$, the next row $R_1=R+1$ is tried until too many states (over $X$) are generated. In that case the results of the previous $R_e$ is kept. It may be the case that in this process the states generated for some row $R$ is smaller than those generated for its predecessor, $R-1$. This means that generation is on a downslope. Downslope bounding will make state generation stop and discard this set of states and default to using the previous.
 
`STATE_GENERATION_LIMIT_PLAY_FACTOR` State generation may be asked for a pool of $X$ many states to be generated. Some times, the algorithm employed to generate states will go over $X$ a bit,  meaning that the pool it generated must be discarded and the previous one must be kept. However, since the number of states generated by locking two subsequent rows is usually quite far apart, it may be beneficial to keep the slightly larger pool. This factor dictates what percentage over the limit $X$ is acceptable. 

`ENABLE_THREADED_STATE_GENERATION` If defined, state generation is performed by parallel threads. This requires libpthread to be linked. For windows hosts, pthread4w (https://sourceforge.net/projects/pthreads4w/) can be compiled and linked. It is advised to select statically linked library files for this tool. 
 
`SMEM_SIZE` defines the assumed shared memory size. This typically varies by device and carveout (for devices that support it) but for the purposes of this solver yet, we chose a fixed limit.
 
`WARP_SIZE` defines the number  of threads in a warp. Whilst the extern variable`warpSize` holds the same information, having this number available at compile time is beneficial.

`WARPS_PER_BLOCK` defines the maximum number of warps possible in a 1D thread block. This can be dynamically calculated.

`COMPLETE_KERNEL_BLOCK_THREAD_COUNT` an expression to work out the number of threads in a 1D thread block with the following constraints:
 - The number of threads must be divisible by 32 (complete warps),
 - Each thread must be able to fit its state in shared memory (size of state may vary by compiler),
 - There must be room for 32x 32bit unsigned integers in shared memory.
 
`STATE_GENERATION_POOL_COUNT` defines the initial size of the state pool used in state generation. If multiple threads are involved, each will initially allocate a pool of this many threads. The pool may be re-allocated on demand if its capacity is exceeded.

`THREAD_INDX_1D_GRID` an expression to find the global index of a thread in the grid.

`MAX(a,b)` a macro evaluating to an expression that determines the maximum value of a,b. Intended for integers, likely okay with floating point but beware of `fast-math`

`MIN(a,b)` a macro evaluating to an expression that determines the minimum value of a,b. Intended for integers, likely okay with floating point but beware of `fast-math`

`FAIL_IF(a)` a macro to check the condition a, and fail if a is false. Acts the same way as assert but allows for the handling of the failure to be different (and future porting to device code. Currently only host-side!)

`CHECK_CUDA_ERROR(expr)` a macro to handle CUDA errors. Whilst the handling may not be ideal as it is always the same (print-and-exit), pretty handling of CUDA errors is not built into this tool yet.

### Preprocessor Checks and Validation
The following list of checks/light validation is performed by the preprocessor and may result in warnings/errors:
 - The value of `N` must be in range $0\lt{}N\leq{}32$ 
 - The `__CUDA_ARCH__` the code is compiled for must be greater or equal than the minimum required compute capability

### Compiling `libpthread` for x64 Windows Hosts under MVCC
The following steps are required to build `libpthread` on Windows using the Visual Studio 2022 toolchain. Likely works for other (future) versions.
1. Install the _Windows Universal CRT_
	- Open the Visual Studio installer
	- Modify the installation of Visual Studio
	- Select "Individual components"
	- Find the Microsoft Universal CRT and install it
	- Hope for the best
2. Download pthread4w: https://sourceforge.net/p/pthreads4w/wiki/Home/
3. Extract the sources somewhere. They won't be needed after this process.
4. Start a Microsoft Visual Studio x64 Shell
	- Go to 'Start' >> 'Visual Studio 2022' >> 'x64 Native Tools Command Prompt'
	- If it is not there, go to 'Program files'>>'Microsoft Visual Studio'>>'2022'>>'Community'>>'VC, Auxilary'>>'Build' and start `vcvars64.bat`
5. cd into the directory with pthreads
6. Run `nmake clean VC-static` (to build the static .lib file for VC)
7. Run `nmake clean VC-static-debug` (for the lib with debug symbols)
8. You should now have 2 `.lib` files in the directory which are something like `libpthreadVC3.lib` and `libpthreadVC3d.lib`
9. Set up the solution on Visual Studio:
	1. Right click the project and go to 'Properties'>>'Configuration Properties'>>'Linker'>>'Input'
	2. Edit the "additional dependencies" and add the `.lib` file name (full) from above (if in debug add the d file otherwise the other)
	3. Go to Configuration 'Properties'>>'Linker'>>'General'
	4. Edit "Aditional library Directories". Add the path to the directory that contains the aforementioned lib files (best move them to a project dir but anything works)
	5. Go to Configuration 'Properties'>>'C/C++'>>'General'
	6. Edit "Additional Include Directories" and add the directory to where the `pthread` header files live. These will be in the directory the lib files are in as well, but you can too move these to a different place and add that directory.
	7. Click on Apply, Ok, Build.

## Further Reading
[1] O. Kullmann, "The OKlibrary: Introducing a "holistic" research platform for (generalised) SAT solving,"  Studies in Logic, vol. 2, no. 1, pp. 20–53, 2009.