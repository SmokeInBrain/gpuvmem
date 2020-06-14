/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */
#include "functions.cuh"

namespace cg = cooperative_groups;

extern long M, N;
extern int iterations, iterthreadsVectorNN, blocksVectorNN, nopositivity, image_count, \
           status_mod_in, flag_opt, verbose_flag, clip_flag, num_gpus, selected, iter, multigpu, firstgpu, reg_term, save_model_input, apply_noise, print_images, gridding;

extern cufftHandle plan1GPU;
extern cufftComplex *device_V, *device_fg_image, *device_I_nu;
extern float *device_I;

extern float *device_dphi, *device_S, *device_dchi2_total, *device_dS, *device_noise_image;
extern float noise_jypix, fg_scale, noise_cut, MINPIX, \
             minpix, lambda, ftol, random_probability, final_chi2, final_S, eta;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, b_noise_aux;
extern float *initial_values, *penalizators, robust_param;
extern double ra, dec, DELTAX, DELTAY, deltau, deltav, crpix1, crpix2;
extern float threshold;
extern float nu_0;
extern int nPenalizators, print_errors, nMeasurementSets, max_number_vis;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern MSDataset *datasets;

extern varsPerGPU *vars_gpu;

typedef float (*FnPtr)(float, float, float, float);

__device__ FnPtr beam_maps[2] = {
        AiryDiskBeam, GaussianBeam
};


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
        __device__ inline operator       T *()
        {
                extern __shared__ int __smem[];
                return (T *)__smem;
        }

        __device__ inline operator const T *() const
        {
                extern __shared__ int __smem[];
                return (T *)__smem;
        }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
        __device__ inline operator       double *()
        {
                extern __shared__ double __smem_d[];
                return (double *)__smem_d;
        }

        __device__ inline operator const double *() const
        {
                extern __shared__ double __smem_d[];
                return (double *)__smem_d;
        }
};

__host__ void goToError()
{
        if(num_gpus > 1) {
                for(int i=firstgpu+1; i<firstgpu + num_gpus; i++) {
                        cudaSetDevice(firstgpu);
                        cudaDeviceDisablePeerAccess(i);
                        cudaSetDevice(i);
                        cudaDeviceDisablePeerAccess(firstgpu);
                }

                for(int i=0; i<num_gpus; i++ ) {
                        cudaSetDevice((i%num_gpus) + firstgpu);
                        cudaDeviceReset();
                }
        }

        printf("An error has ocurred, exiting\n");
        exit(0);

}

__host__ long NearestPowerOf2(long x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
}


bool isPow2(unsigned int x)
{
        return ((x&(x-1))==0);
}

__host__ void readInputDat(char *file)
{
        FILE *fp;
        char item[50];
        char status[50];
        if((fp = fopen(file, "r")) == NULL) {
                printf("ERROR. The input file wasn't provided by the user.\n");
                goToError();
        }else{
                while(true) {
                        int ret = fscanf(fp, "%s %s", item, status);

                        if(ret==EOF) {
                                break;
                        }else{
                                if (strcmp(item,"noise_cut")==0) {
                                        if(noise_cut == -1) {
                                                noise_cut = atof(status);
                                        }
                                }else if(strcmp(item,"ftol")==0) {
                                        ftol = atof(status);
                                } else if(strcmp(item,"random_probability")==0) {
                                        if(random_probability == -1) {
                                                random_probability = atof(status);
                                        }
                                }else{
                                        printf("Keyword not defined in input\n");
                                        goToError();
                                }
                        }
                }
        }
}

__host__ void print_help() {
        printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
        printf("    -h  --help             Shows this\n");
        printf( "   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2)\n");
        printf( "   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2)\n");
        printf( "   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)\n");
        printf( "   -i  --input            The name of the input file/s (separated by a comma) of visibilities(MS)\n");
        printf( "   -o  --output           The name of the output file/s (separated by a comma) of residual visibilities(MS)\n");
        printf( "   -O  --output-image     The name of the output image FITS file\n");
        printf("    -I  --inputdat         The name of the input file of parameters\n");
        printf("    -m  --model_input      FITS file including a complete header for astrometry\n");
        printf("    -n  --noise            Noise Parameter (Optional)\n");
        printf("    -N  --noise-cut        Noise-cut Parameter (Optional)\n");
        printf("    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)\n");
        printf("    -e  --eta              Variable that controls the minimum image value (Default eta = -1.0)\n");
        printf("    -p  --path             MEM path to save FITS images. With last / included. (Example ./../mem/)\n");
        printf("    -f  --file             Output file where final objective function values are saved (Optional)\n");
        printf("    -M  --multigpu         Number of GPUs to use multiGPU image synthesis (Default OFF => 0)\n");
        printf("    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
        printf("    -t  --iterations       Number of iterations for optimization (Default = 500)\n");
        printf("    -g  --gridding         Use gridding to decrease the number of visibilities. This is done in CPU (Need to select the CPU threads that will grid the input visibilities)\n");
        printf("    -F  --nu_0             Reference frequency in Hz (if alpha is not zero)");
        printf("    -z  --initial_values   Initial conditions for image/s\n");
        printf("    -Z  --penalizators     penalizators for Fi\n");
        printf("    -R  --robust-parameter Robust weighting parameter when gridding. -2.0 for uniform weighting, 2.0 for natural weighting and 0.0 for a tradeoff between these two. (Default R = 2.0).\n");
        printf("    -T  --threshold        Threshold to calculate the spectral index image from a certain number of sigmas in I_nu_0\n");
        printf("    -c  --copyright        Shows copyright conditions\n");
        printf("    -w  --warranty         Shows no warranty details\n");
        printf("        --savemodel-input  Saves the model visibilities on the model column of the input MS\n");
        printf("        --nopositivity     Run gpuvmem using chi2 with no posititivy restriction\n");
        printf("        --apply-noise      Apply random gaussian noise to visibilities\n");
        printf("        --clipping         Clips the image to positive values\n");
        printf("        --print-images     Prints images per iteration\n");
        printf("        --print-errors     Prints final error images\n");
        printf("        --verbose          Shows information through all the execution\n");
}

__host__ char *strip(const char *string, const char *chars)
{
        char * newstr = (char*)malloc(strlen(string) + 1);
        int counter = 0;

        for (; *string; string++) {
                if (!strchr(chars, *string)) {
                        newstr[ counter ] = *string;
                        ++counter;
                }
        }

        newstr[counter] = 0;
        return newstr;
}

__host__ Vars getOptions(int argc, char **argv) {
        Vars variables;
        variables.multigpu = "NULL";
        variables.ofile = "NULL";
        variables.path = "mem/";
        variables.output_image = "mod_out.fits";
        variables.select = 0;
        variables.initial_values = "NULL";
        variables.penalization_factors = "NULL";
        variables.blockSizeX = -1;
        variables.blockSizeY = -1;
        variables.blockSizeV = -1;
        variables.it_max = 500;
        variables.noise = -1;
        variables.randoms = -1.0;
        variables.noise_cut = -1;
        variables.eta = -1.0;
        variables.gridding = 0;
        variables.robust_param = 2.0;
        variables.nu_0 = -1;
        variables.threshold = 0.0;


        long next_op;
        const char* const short_op = "hcwi:o:O:I:m:n:N:r:R:f:M:s:e:p:X:Y:V:t:g:z:T:F:Z:";

        const struct option long_op[] = { //Flag for help, copyright and warranty
                {"help", 0, NULL, 'h' },
                {"warranty", 0, NULL, 'w' },
                {"copyright", 0, NULL, 'c' },
                /* These options set a flag. */
                {"verbose", 0, &verbose_flag, 1},
                {"nopositivity", 0, &nopositivity, 1},
                {"clipping", 0, &clip_flag, 1},
                {"apply-noise", 0, &apply_noise, 1},
                {"print-images", 0, &print_images, 1},
                {"print-errors", 0, &print_errors, 1},
                {"savemodel-input", 0, &save_model_input, 1},
                /* These options don’t set a flag. */
                {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"output-image", 1, NULL, 'O'},
                {"threshold", 0, NULL, 'T'}, {"nu_0", 0, NULL, 'F'},
                {"inputdat", 1, NULL, 'I'}, {"model_input", 1, NULL, 'm' }, {"noise", 0, NULL, 'n' },
                {"multigpu", 0, NULL, 'M'}, {"select", 1, NULL, 's'},
                {"path", 1, NULL, 'p'}, {"robust-parameter", 0, NULL, 'R'}, {"eta", 0, NULL, 'e'},
                {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'},
                {"iterations", 0, NULL, 't'}, {"noise-cut", 0, NULL, 'N' }, {"initial_values", 1, NULL, 'z'}, {"penalizators", 0, NULL, 'Z'},
                {"randoms", 0, NULL, 'r' }, {"file", 0, NULL, 'f' }, {"gridding", 0, NULL, 'g' }, { NULL, 0, NULL, 0 }
        };

        if (argc == 1) {
                printf(
                        "ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
                print_help();
                exit(EXIT_SUCCESS);
        }
        int option_index = 0;
        while (1) {
                next_op = getopt_long(argc, argv, short_op, long_op, &option_index);
                if (next_op == -1) {
                        break;
                }

                switch (next_op) {
                case 0:
                        /* If this option set a flag, do nothing else now. */
                        if (long_op[option_index].flag != 0)
                                break;
                        printf ("option %s", long_op[option_index].name);
                        if (optarg)
                                printf (" with arg %s", optarg);
                        printf ("\n");
                        break;
                case 'h':
                        print_help();
                        exit(EXIT_SUCCESS);
                case 'w':
                        print_warranty();
                        exit(EXIT_SUCCESS);
                case 'c':
                        print_copyright();
                        exit(EXIT_SUCCESS);
                case 'i':
                        variables.input = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.input, optarg);
                        break;
                case 'o':
                        variables.output = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output, optarg);
                        break;
                case 'O':
                        variables.output_image = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output_image, optarg);
                        break;
                case 'I':
                        variables.inputdat = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.inputdat, optarg);
                        break;
                case 'm':
                        variables.modin = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.modin, optarg);
                        break;
                case 'n':
                        variables.noise = atof(optarg);
                        break;
                case 'e':
                        variables.eta = atof(optarg);
                        break;
                case 'N':
                        variables.noise_cut = atof(optarg);
                        break;
                case 'F':
                        variables.nu_0 = atof(optarg);
                        break;
                case 'T':
                        variables.threshold = atof(optarg);
                        break;
                case 'p':
                        variables.path = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.path, optarg);
                        break;
                case 'M':
                        variables.multigpu = optarg;
                        break;
                case 'r':
                        variables.randoms = atof(optarg);
                        break;
                case 'R':
                        variables.robust_param = atof(optarg);
                        break;
                case 'f':
                        variables.ofile = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.ofile, optarg);
                        break;
                case 's':
                        variables.select = atoi(optarg);
                        break;
                case 'X':
                        variables.blockSizeX = atoi(optarg);
                        break;
                case 'Y':
                        variables.blockSizeY = atoi(optarg);
                        break;
                case 'V':
                        variables.blockSizeV = atoi(optarg);
                        break;
                case 't':
                        variables.it_max = atoi(optarg);
                        break;
                case 'g':
                        variables.gridding = atoi(optarg);
                        break;
                case 'z':
                        variables.initial_values = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.initial_values, optarg);
                        break;
                case 'Z':
                        variables.penalization_factors = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.penalization_factors, optarg);
                        break;
                case '?':
                        print_help();
                        exit(EXIT_FAILURE);
                case -1:
                        break;
                default:
                        print_help();
                        exit(EXIT_FAILURE);
                }
        }

        if(variables.blockSizeX == -1 && variables.blockSizeY == -1 && variables.blockSizeV == -1 ||
           strcmp(strip(variables.input, " "),"") == 0 && strcmp(strip(variables.output, " "),"") == 0 && strcmp(strip(variables.output_image, " "),"") == 0 && strcmp(strip(variables.inputdat, " "),"") == 0 ||
           strcmp(strip(variables.modin, " "),"") == 0 && strcmp(strip(variables.path, " "),"") == 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.randoms > 1.0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.gridding < 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(strcmp(variables.multigpu,"NULL")!=0 && variables.select != 0) {
                print_help();
                exit(EXIT_FAILURE);
        }
        return variables;
}



#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

__host__ void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

        //get device capability, to avoid block/grid size exceed the upper bound
        cudaDeviceProp prop;
        int device;
        gpuErrchk(cudaGetDevice(&device));
        gpuErrchk(cudaGetDeviceProperties(&prop, device));


        threads = (n < maxThreads*2) ? NearestPowerOf2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);

        if (blocks > prop.maxGridSize[0])
        {
                printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                       blocks, prop.maxGridSize[0], threads*2, threads);

                blocks /= 2;
                threads *= 2;
        }

        blocks = MIN(maxBlocks, blocks);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void deviceReduceKernel(T *g_idata, T *g_odata, unsigned int n)
{
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        T *sdata = SharedMemory<T>();

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
        unsigned int gridSize = blockSize*2*gridDim.x;

        T mySum = 0;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
                mySum += g_idata[i];

                // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
                if (nIsPow2 || i + blockSize < n)
                        mySum += g_idata[i+blockSize];

                i += gridSize;
        }

        // each thread puts its local sum into shared memory
        sdata[tid] = mySum;
        cg::sync(cta);


        // do reduction in shared mem
        if ((blockSize >= 512) && (tid < 256))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) &&(tid < 128))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid <  64))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
        if ( tid < 32 )
        {
                cg::coalesced_group active = cg::coalesced_threads();

                // Fetch final intermediate sum from 2nd warp
                if (blockSize >=  64) mySum += sdata[tid + 32];
                // Reduce final warp using shuffle
                for (int offset = warpSize/2; offset > 0; offset /= 2)
                {
                        mySum += active.shfl_down(mySum, offset);
                }
        }
#else
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        cg::sync(cta);

        if ((blockSize >=  32) && (tid < 16))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        cg::sync(cta);

        if ((blockSize >=  16) && (tid <  8))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        cg::sync(cta);

        if ((blockSize >=   8) && (tid <  4))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        cg::sync(cta);

        if ((blockSize >=   4) && (tid <  2))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        cg::sync(cta);

        if ((blockSize >=   2) && ( tid <  1))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  1];
        }

        cg::sync(cta);
#endif

        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = mySum;
}





template <class T>
__host__ T deviceReduce(T *in, long N)
{
        T *device_out;

        int maxThreads = 256;
        int maxBlocks = NearestPowerOf2(N)/maxThreads;

        int threads = 0;
        int blocks = 0;

        getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads);

        //printf("N %d, threads: %d, blocks %d\n", N, threads, blocks);
        //threads = maxThreads;
        //blocks = NearestPowerOf2(N)/threads;

        gpuErrchk(cudaMalloc(&device_out, sizeof(T)*blocks));
        gpuErrchk(cudaMemset(device_out, 0, sizeof(T)*blocks));

        int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

        bool isPower2 = isPow2(N);

        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);

        if(isPower2) {
                switch (threads) {
                case 512:
                        deviceReduceKernel<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 256:
                        deviceReduceKernel<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 128:
                        deviceReduceKernel<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 64:
                        deviceReduceKernel<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 32:
                        deviceReduceKernel<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 16:
                        deviceReduceKernel<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 8:
                        deviceReduceKernel<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 4:
                        deviceReduceKernel<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 2:
                        deviceReduceKernel<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 1:
                        deviceReduceKernel<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                }
        }else{
                switch (threads) {
                case 512:
                        deviceReduceKernel<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 256:
                        deviceReduceKernel<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 128:
                        deviceReduceKernel<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 64:
                        deviceReduceKernel<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 32:
                        deviceReduceKernel<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 16:
                        deviceReduceKernel<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 8:
                        deviceReduceKernel<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 4:
                        deviceReduceKernel<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 2:
                        deviceReduceKernel<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 1:
                        deviceReduceKernel<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                }
        }

        T *h_odata = (T *) malloc(blocks*sizeof(T));
        T sum = 0;

        gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(T),cudaMemcpyDeviceToHost));
        for (int i=0; i<blocks; i++)
        {
                sum += h_odata[i];
        }
        cudaFree(device_out);
        free(h_odata);
        return sum;
}

__global__ void fftshift_2D(cufftComplex *data, int N1, int N2)
{
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        int j = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < N1 && j < N2) {
                float a = 1-2*((i+j)&1);

                data[N2*i+j].x *= a;
                data[N2*i+j].y *= a;
        }
}

__global__ void DFT2D(cufftComplex *Vm, cufftComplex *I, double3 *UVW, float *noise, float noise_cut, float xobs, float yobs, double DELTAX, double DELTAY, int M, int N, int numVisibilities)
{
        int v = threadIdx.x + blockDim.x * blockIdx.x;

        if (v < numVisibilities) {
                int x0, y0;
                double x, y, z;
                float cosk, sink, Ukv, Vkv, Wkv, I_sky;
                cufftComplex Vmodel;
                Vmodel.x = 0.0f;
                Vmodel.y = 0.0f;
                double3 uvw = UVW[v];
                for(int i=0; i<M; i++) {
                        for(int j=0; j<N; j++) {
                                x0 = xobs;
                                y0 = yobs;
                                x = (j - x0) * DELTAX * RPDEG_D;
                                y = (i - y0) * DELTAY * RPDEG_D;
                                z = sqrtf(1-x*x-y*y)-1;
                                I_sky = I[N*i+j].x;
                                if(noise[N*i+j] > noise_cut) {
                                        Ukv = x * uvw.x;
                                        Vkv = y * uvw.y;
                                        Wkv = z * uvw.z;
                    #if (__CUDA_ARCH__ >= 300 )
                                        sincospif(2.0*(Ukv+Vkv+Wkv), &sink, &cosk);
                    #else
                                        cosk = cospif(2.0*(Ukv+Vkv+Wkv));
                                        sink = sinpif(2.0*(Ukv+Vkv+Wkv));
                    #endif
                                        Vmodel.x +=  I_sky * cosk;
                                        Vmodel.y += -I_sky * sink;
                                }
                        }
                }

                Vm[v].x = Vmodel.x;
                Vm[v].y = Vmodel.y;
        }
}

__host__ void do_gridding(std::vector<Field>& fields, MSData *data, double deltau, double deltav, int M, int N, float robust)
{
        std::vector<float> g_weights(M*N);
        std::vector<cufftComplex> g_Vo(M*N);
        std::vector<double3> g_uvw(M*N);
        std::vector<float> S(M*N);
        cufftComplex complex_zero;
        complex_zero.x = 0.0f;
        complex_zero.y = 0.0f;

        double3 double3_zero;
        double3_zero.x = 0.0;
        double3_zero.y = 0.0;
        double3_zero.z = 0.0;

        int local_max = 0;
        int max = 0;
        float pow2_factor, S2, w_avg;
        for(int f=0; f < data->nfields; f++) {
                for(int i=0; i < data->total_frequencies; i++) {
                        for(int s=0; s< data->nstokes; s++) {
                                fields[f].backup_visibilities[i][s].uvw.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                fields[f].backup_visibilities[i][s].weight.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                fields[f].backup_visibilities[i][s].Vo.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                        #pragma omp parallel for schedule(static, 1)
                                for (int z = 0; z < fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++) {

                                        int j, k;
                                        double3 uvw;
                                        float w;
                                        cufftComplex Vo;

                                        uvw = fields[f].visibilities[i][s].uvw[z];
                                        w = fields[f].visibilities[i][s].weight[z];
                                        Vo = fields[f].visibilities[i][s].Vo[z];

                                        //Backing up original visibilities and (u,v) positions
                                        fields[f].backup_visibilities[i][s].uvw[z] = uvw;
                                        fields[f].backup_visibilities[i][s].Vo[z] = Vo;
                                        fields[f].backup_visibilities[i][s].weight[z] = w;

                                        // Visibilities from metres to klambda
                                        uvw.x *= fields[f].nu[i] / LIGHTSPEED;
                                        uvw.y *= fields[f].nu[i] / LIGHTSPEED;
                                        uvw.z *= fields[f].nu[i] / LIGHTSPEED;

                                        //Apply hermitian symmetry (it will be applied afterwards)
                                        if (uvw.x < 0.0) {
                                                uvw.x *= -1.0;
                                                uvw.y *= -1.0;
                                                Vo.y *= -1.0;
                                        }

                                        j = round(uvw.x / fabs(deltau) + N / 2);
                                        k = round(uvw.y / fabs(deltav) + M / 2);

                                        if (k < M && j < N) {
                                #pragma omp critical
                                                {
                                                        g_Vo[N * k + j].x += w * Vo.x;
                                                        g_Vo[N * k + j].y += w * Vo.y;
                                                        g_weights[N * k + j] += w;
                                                        S[N * k + j] = 1;
                                                }
                                        }
                                }

                                int visCounter = 0;
                                float gridWeightSum = 0.0f;

                                for (int k = 0; k < M; k++) {
                                        for (int j = 0; j < N; j++) {
                                                float weight = g_weights[N * k + j];
                                                if (weight > 0.0f) {
                                                        gridWeightSum += weight;
                                                        visCounter++;
                                                }
                                        }
                                }

                                // Briggs/Robust formula
                                pow2_factor = pow(10.0, -2.0 * robust);
                                w_avg = gridWeightSum / visCounter;
                                S2 = 5.0f * 5.0f * pow2_factor / w_avg;

                        #pragma omp parallel for schedule(static, 1)
                                for (int k = 0; k < M; k++) {
                                        for (int j = 0; j < N; j++) {
                                                double deltau_meters = fabs(deltau) * (LIGHTSPEED / fields[f].nu[i]);
                                                double deltav_meters = fabs(deltav) * (LIGHTSPEED / fields[f].nu[i]);

                                                double u_meters = (j - (N / 2)) * deltau_meters;
                                                double v_meters = (k - (M / 2)) * deltav_meters;

                                                g_uvw[N * k + j].x = u_meters;
                                                g_uvw[N * k + j].y = v_meters;

                                                float weight = g_weights[N * k + j];
                                                if (weight > 0.0f) {
                                                        g_Vo[N * k + j].x /= weight;
                                                        g_Vo[N * k + j].y /= weight;
                                                        g_weights[N * k + j] /= (1 + weight * S2);
                                                } else {
                                                        g_weights[N * k + j] = 0.0f;
                                                }
                                        }
                                }

                                fields[f].visibilities[i][s].uvw.resize(visCounter);
                                fields[f].visibilities[i][s].Vo.resize(visCounter);

                                cufftComplex zero;

                                fields[f].visibilities[i][s].Vm.resize(visCounter);
                                memset(&fields[f].visibilities[i][s].Vm[0], 0, fields[f].visibilities[i][s].Vm.size() * sizeof(cufftComplex));

                                fields[f].visibilities[i][s].weight.resize(visCounter);

                                int l = 0;
                                for (int k = 0; k < M; k++) {
                                        for (int j = 0; j < N; j++) {
                                                float weight = g_weights[N * k + j];
                                                if (weight > 0.0f) {
                                                        fields[f].visibilities[i][s].uvw[l].x = g_uvw[N * k + j].x;
                                                        fields[f].visibilities[i][s].uvw[l].y = g_uvw[N * k + j].y;
                                                        fields[f].visibilities[i][s].Vo[l].x = g_Vo[N * k + j].x;
                                                        fields[f].visibilities[i][s].Vo[l].y = g_Vo[N * k + j].y;
                                                        fields[f].visibilities[i][s].weight[l] = g_weights[N * k + j];
                                                        l++;
                                                }
                                        }
                                }


                                fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s] = fields[f].numVisibilitiesPerFreqPerStoke[i][s];

                                if (fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
                                        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = visCounter;
                                }else{
                                        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = 0;
                                }

                                std::fill_n(g_weights.begin(), M*N, 0.0f);
                                std::fill_n(g_uvw.begin(), M*N,double3_zero);
                                std::fill_n(g_Vo.begin(), M*N, complex_zero);

                        }

                        local_max = *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i].data(),fields[f].numVisibilitiesPerFreqPerStoke[i].data()+data->nstokes);
                        if(local_max > max) {
                                max = local_max;
                        }

                        fields[f].backup_numVisibilitiesPerFreq[i] = fields[f].numVisibilitiesPerFreq[i];
                }
        }


        data->max_number_visibilities_in_channel_and_stokes = max;
}


__host__ float calculateNoise(std::vector<MSDataset>& datasets, int *total_visibilities, int blockSizeV, int gridding)
{
        //Declaring block size and number of blocks for visibilities
        float variance;
        float sum_weights = 0.0;
        long UVpow2;
        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        for(int i=0; i< datasets[d].data.total_frequencies; i++) {
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        //Calculating beam noise
                                        for (int j = 0; j < datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; j++) {
                                                if (datasets[d].fields[f].visibilities[i][s].weight[j] > 0.0) {
                                                        //sum_inverse_weight += 1 / fields[f].visibilities[i][s].weight[j];
                                                        sum_weights += datasets[d].fields[f].visibilities[i][s].weight[j];
                                                }
                                        }
                                        *total_visibilities += datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s];
                                        UVpow2 = NearestPowerOf2(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
                                        datasets[d].fields[f].device_visibilities[i][s].numBlocksUV = UVpow2 / datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV;
                                }
                        }
                }
        }



        variance = 1.0f/sum_weights;

        if(verbose_flag) {
                float aux_noise = sqrt(variance);
                printf("Calculated NOISE %e\n", aux_noise);
        }

        if(beam_noise == -1 || gridding > 0)
        {
                beam_noise = sqrt(variance);
                if(verbose_flag) {
                        printf("No NOISE keyword detected in header or you might be using gridding\n");
                        printf("Using NOISE: %e ...\n", beam_noise);
                }
        }else{
                printf("Using header keyword NOISE anyway...\n");
                printf("Keyword NOISE = %e\n", beam_noise);
        }

        return sum_weights;
}

__host__ void griddedTogrid(std::vector<cufftComplex>& Vm_gridded, std::vector<cufftComplex> Vm_gridded_sp, std::vector<double3> uvw_gridded_sp, double deltau, double deltav, float freq, long M, long N, int numvis)
{

        double deltau_meters = fabs(deltau) * (LIGHTSPEED/freq);
        double deltav_meters = fabs(deltav) * (LIGHTSPEED/freq);

        cufftComplex complex_zero;
        complex_zero.x = 0.0f;
        complex_zero.y = 0.0f;

        std::fill_n(Vm_gridded.begin(), M*N, complex_zero);

        int j, k;
        for(int i=0; i<numvis; i++) {
                j = (uvw_gridded_sp[i].x / deltau_meters) + N/2;
                k = (uvw_gridded_sp[i].y / deltav_meters) + M/2;
                Vm_gridded[N*k+j] = Vm_gridded_sp[i];
        }
}

__host__ void degridding(std::vector<Field>& fields, MSData data, double deltau, double deltav, int num_gpus, int firstgpu, int blockSizeV, long M, long N, int gridding)
{

        long UVpow2;

        residualsToHost(fields, data, num_gpus, firstgpu);


        int nthreads = gridding;

        std::vector<std::vector<cufftComplex> > gridded_visibilities(nthreads, std::vector<cufftComplex> (M*N));

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                for(int s=0; s<data.nstokes; s++) {
                                        // Put gridded visibilities in a M*N grid
                                        griddedTogrid(gridded_visibilities[0], fields[f].visibilities[i][s].Vm,
                                                      fields[f].visibilities[i][s].uvw, deltau, deltav, fields[f].nu[i], M, N,
                                                      fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                        /*
                                           Model visibilities and original (u,v) positions to GPU.
                                         */

                                        // Now the number of visibilities will be the original one.

                                        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s];

                                        //  We allocate memory for arrays using the original number of visibilities
                                        fields[f].visibilities[i][s].uvw.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].weight.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].Vm.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].Vo.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].uvw,
                                                             sizeof(double3) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].weight,
                                                             sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

                                        // Copy original Vo visibilities to host
                                        fields[f].visibilities[i][s].Vo.assign(fields[f].backup_visibilities[i][s].Vo.begin(), fields[f].backup_visibilities[i][s].Vo.end());

                                        // Copy gridded model visibilities to device
                                        gpuErrchk(cudaMemcpy(vars_gpu[0].device_V, gridded_visibilities[0].data(), sizeof(cufftComplex) * M * N,
                                                             cudaMemcpyHostToDevice));

                                        // Copy original (u,v) positions and weights to host and device
                                        fields[f].visibilities[i][s].uvw.assign(fields[f].backup_visibilities[i][s].uvw.begin(), fields[f].backup_visibilities[i][s].uvw.end());
                                        fields[f].visibilities[i][s].weight.assign(fields[f].backup_visibilities[i][s].weight.begin(), fields[f].backup_visibilities[i][s].weight.end());

                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].uvw, fields[f].backup_visibilities[i][s].uvw.data(),
                                                             sizeof(double3) * fields[f].backup_visibilities[i][s].uvw.size(),
                                                             cudaMemcpyHostToDevice));
                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].Vo, fields[f].backup_visibilities[i][s].Vo.data(),
                                                             sizeof(cufftComplex) * fields[f].backup_visibilities[i][s].Vo.size(),
                                                             cudaMemcpyHostToDevice));
                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].weight, fields[f].backup_visibilities[i][s].weight.data(),
                                                             sizeof(float) * fields[f].backup_visibilities[i][s].weight.size(), cudaMemcpyHostToDevice));

                                        UVpow2 = NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
                                        fields[f].device_visibilities[i][s].numBlocksUV = UVpow2 / fields[f].device_visibilities[i][s].threadsPerBlockUV;

                                        hermitianSymmetry <<< fields[f].device_visibilities[i][s].numBlocksUV,
                                                fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                        (fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].Vo, fields[f].nu[i], fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        cudaDeviceSynchronize();

                                        // Interpolation / Degridding
                                        vis_mod2 <<< fields[f].device_visibilities[i][s].numBlocksUV,
                                                fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                        (fields[f].device_visibilities[i][s].Vm, vars_gpu[0].device_V, fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].weight, deltau, deltav, fields[f].numVisibilitiesPerFreqPerStoke[i][s], N);
                                        cudaDeviceSynchronize();
                                        // Freeing backup arrays

                                        fields[f].backup_visibilities[i][s].uvw.clear();
                                        fields[f].backup_visibilities[i][s].weight.clear();
                                        fields[f].backup_visibilities[i][s].Vo.clear();
                                }

                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
            #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                for(int s=0; s<data.nstokes; s++) {
                                        // Put gridded visibilities in a M*N grid
                                        griddedTogrid(gridded_visibilities[j], fields[f].visibilities[i][s].Vm,
                                                      fields[f].visibilities[i][s].uvw, deltau, deltav, fields[f].nu[i], M, N,
                                                      fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                        /*
                                           Model visibilities and original (u,v) positions to GPU.
                                         */

                                        // Now the number of visibilities will be the original one.
                                        fields[f].numVisibilitiesPerFreqPerStoke[i][s] = fields[f].backup_numVisibilitiesPerFreqPerStoke[i][s];

                                        fields[f].visibilities[i][s].uvw.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].weight.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].Vm.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].visibilities[i][s].Vo.resize(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].Vm,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMemset(fields[f].device_visibilities[i][s].Vm, 0,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].uvw,
                                                             sizeof(double3) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].Vo,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i][s].weight,
                                                             sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s]));

                                        // Copy original Vo visibilities to host
                                        fields[f].visibilities[i][s].Vo.assign(fields[f].backup_visibilities[i][s].Vo.begin(), fields[f].backup_visibilities[i][s].Vo.end());

                                        // Copy gridded model visibilities to device
                                        gpuErrchk(cudaMemcpy(vars_gpu[i % num_gpus].device_V, gridded_visibilities[j].data(),
                                                             sizeof(cufftComplex) * M * N, cudaMemcpyHostToDevice));

                                        // Copy original (u,v) positions and weights to host and device

                                        fields[f].visibilities[i][s].uvw.assign(fields[f].backup_visibilities[i][s].uvw.begin(), fields[f].backup_visibilities[i][s].uvw.end());
                                        fields[f].visibilities[i][s].weight.assign(fields[f].backup_visibilities[i][s].weight.begin(), fields[f].backup_visibilities[i][s].weight.end());

                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].uvw, fields[f].backup_visibilities[i][s].uvw.data(),
                                                             sizeof(double3) * fields[f].backup_visibilities[i][s].uvw.size(),
                                                             cudaMemcpyHostToDevice));
                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].Vo, fields[f].backup_visibilities[i][s].Vo.data(),
                                                             sizeof(cufftComplex) * fields[f].backup_visibilities[i][s].Vo.size(),
                                                             cudaMemcpyHostToDevice));
                                        gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i][s].weight, fields[f].backup_visibilities[i][s].weight.data(),
                                                             sizeof(float) * fields[f].backup_visibilities[i][s].weight.size(), cudaMemcpyHostToDevice));

                                        UVpow2 = NearestPowerOf2(fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        fields[f].device_visibilities[i][s].threadsPerBlockUV = blockSizeV;
                                        fields[f].device_visibilities[i][s].numBlocksUV = UVpow2 / fields[f].device_visibilities[i][s].threadsPerBlockUV;

                                        hermitianSymmetry <<< fields[f].device_visibilities[i][s].numBlocksUV,
                                                fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                        (fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].Vo, fields[f].nu[i], fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        cudaDeviceSynchronize();

                                        // Interpolation / Degridding
                                        vis_mod2 <<< fields[f].device_visibilities[i][s].numBlocksUV,
                                                fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                        (fields[f].device_visibilities[i][s].Vm, vars_gpu[i % num_gpus].device_V, fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].weight, deltau, deltav, fields[f].numVisibilitiesPerFreqPerStoke[i][s], N);
                                        cudaDeviceSynchronize();

                                        // Freeing backup arrays

                                        fields[f].backup_visibilities[i][s].uvw.clear();
                                        fields[f].backup_visibilities[i][s].weight.clear();
                                        fields[f].backup_visibilities[i][s].Vo.clear();
                                }
                        }
                }
        }


}

__host__ void initFFT(varsPerGPU *vars_gpu, long M, long N, int firstgpu, int num_gpus)
{

        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                if ((cufftPlan2d(&vars_gpu[g].plan, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
                        printf("cufft plan error\n");
                        exit(-1);
                }
        }
}

__host__ void FFT2D(cufftComplex *V, cufftComplex *I, cufftHandle plan, int M, int N, bool shift)
{

        if(shift) {
                fftshift_2D << < numBlocksNN, threadsPerBlockNN >> > (I, M, N);
                gpuErrchk(cudaDeviceSynchronize());
        }

        if ((cufftExecC2C(plan,
                          (cufftComplex *) I,
                          (cufftComplex *) V,
                          CUFFT_FORWARD)) != CUFFT_SUCCESS) {
                printf("CUFFT exec error\n");
                goToError();
        }
        gpuErrchk(cudaDeviceSynchronize());

        if(shift) {
                fftshift_2D << < numBlocksNN, threadsPerBlockNN >> > (V, M, N);
                gpuErrchk(cudaDeviceSynchronize());

        }


}

/*__global__ void do_gridding(float *u, float *v, cufftComplex *Vo, cufftComplex *Vo_g, float *w, float *w_g, int* count, float deltau, float deltav, int visibilities, int M, int N)
   {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < visibilities)
   {
    int k, j;
    j = roundf(u[i]/deltau + M/2);
    k = roundf(v[i]/deltav + N/2);

    if (k < M && j < N)
    {

      atomicAdd(&Vo_g[N*k+j].x, Vo[i].x);
      atomicAdd(&Vo_g[N*k+j].y, Vo[i].y);
      atomicAdd(&w_g[N*k+j], (1.0/w[i]));
      atomicAdd(&count[N*k*j], 1);
    }
   }
   }

   __global__ void calculateCoordinates(float *u_g, float *v_g, float deltau, float deltav, int M, int N)
   {
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   u_g[N*i+j] = j*deltau - (N/2)*deltau;
   v_g[N*i+j] = i*deltav - (N/2)*deltav;
   }

   __global__ void calculateAvgVar(cufftComplex *V_g, float *w_g, int *count, int M, int N)
   {
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   int counter = count[N*i+j];
   if(counter > 0){
    V_g[N*i+j].x = V_g[N*i+j].x / counter;
    V_g[N*i+j].y = V_g[N*i+j].y / counter;
    w_g[N*i+j] = counter / w_g[N*i+j];
   }else{
    w_g[N*i+j] = 0.0;
   }
   }*/

__global__ void hermitianSymmetry(double3 *UVW, cufftComplex *Vo, float freq, int numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                if(UVW[i].x < 0.0) {
                        UVW[i].x *= -1.0;
                        UVW[i].y *= -1.0;
                        Vo[i].y *= -1.0;
                }
                UVW[i].x *= freq / LIGHTSPEED;
                UVW[i].y *= freq / LIGHTSPEED;
                UVW[i].z *= freq / LIGHTSPEED;
        }
}


__device__ float AiryDiskBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float atten;
        float r = pb_factor * lambda / antenna_diameter;
        float bessel_arg = PI*distance/(r/RZ);
        float bessel_func = j1f(bessel_arg);
        if(distance == 0.0f) {
                atten = 1.0f;
        }else{
                atten = 4.0f * (bessel_func/bessel_arg) * (bessel_func/bessel_arg);
        }
        return atten;
}

__device__ float GaussianBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float fwhm = pb_factor * lambda / antenna_diameter;
        float c = 4.0*logf(2.0);
        float r = distance/fwhm;
        float atten = expf(-c*r*r);
        return atten;
}


__device__ float attenuation(float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY, int primary_beam)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten_result, atten;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG_D;
        float y = (i - y0) * DELTAY * RPDEG_D;

        float arc = sqrtf(x*x+y*y);
        float lambda = LIGHTSPEED/freq;

        atten = (*beam_maps[primary_beam])(arc, lambda, antenna_diameter, pb_factor);

        if(arc <= pb_cutoff) {
                atten_result = atten;
        }else{
                atten_result = 0.0f;
        }

        return atten_result;
}

__device__ cufftComplex WKernel(double w, float xobs, float yobs, double DELTAX, double DELTAY)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        cufftComplex Wk;
        float cosk, sink;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG_D;
        float y = (i - y0) * DELTAY * RPDEG_D;
        float z = sqrtf(1-x*x-y*y)-1;
        float arg = 2.0f*w*z;

    #if (__CUDA_ARCH__ >= 300 )
        sincospif(arg, &sink, &cosk);
    #else
        cosk = cospif(arg);
        sink = sinpif(arg);
    #endif

        Wk.x = cosk;
        Wk.y = -sink;


}
__global__ void total_attenuation(float *total_atten, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY, long N, int primary_beam)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY, primary_beam);

        total_atten[N*i+j] += attenPerFreq;
}

__global__ void mean_attenuation(float *total_atten, int channels, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        total_atten[N*i+j] /= channels;
}


__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = total_atten[N*i+j];
        weight_image[N*i+j] += (atten / noise_jypix) * (atten / noise_jypix);
}

__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float noiseval;
        noiseval = sqrtf(1.0/weight_image[N*i+j]);
        noise_image[N*i+j] = noiseval;
}

__global__ void apply_beam2I(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *image, long N, float xobs, float yobs, float fg_scale, float freq, double DELTAX, double DELTAY, int primary_beam)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY, primary_beam);

        image[N*i+j].x = image[N*i+j].x * atten * fg_scale;
        //image[N*i+j].x = image[N*i+j].x * atten;
        image[N*i+j].y = 0.0;
}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, double xphs, double yphs)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float u,v, phase, c, s;
        cufftComplex cphase;
        double upix = xphs/(double)M;
        double vpix = yphs/(double)N;

        if(j < M/2) {
                u = upix * j;
        }else{
                u = upix * (j-M);
        }

        if(i < N/2) {
                v = vpix * i;
        }else{
                v = vpix * (i-N);
        }

        phase = 2.0*(u+v);
    #if (__CUDA_ARCH__ >= 300 )
        sincospif(phase, &s, &c);
    #else
        c = cospif(phase);
        s = sinpif(phase);
    #endif

        cphase.x = c;
        cphase.y = s;
        data[N*i+j] = multComplexComplex(data[N*i+j], cphase);
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, double3 *UVW, float *weight, double deltau, double deltav, long numVisibilities, long N)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int i1, i2, j1, j2;
        double du, dv;
        double2 uv;
        cufftComplex v11, v12, v21, v22;
        float Zreal;
        float Zimag;

        if (i < numVisibilities) {

                uv.x = UVW[i].x/fabs(deltau);
                uv.y = UVW[i].y/deltav;

                if (fabs(uv.x) <= (N/2)+0.5 && fabs(uv.y) <= (N/2)+0.5) {

                        if(uv.x < 0.0)
                                uv.x = round(uv.x+N);


                        if(uv.y < 0.0)
                                uv.y = round(uv.y+N);


                        i1 = (int)uv.x;
                        i2 = (i1+1)%N;
                        du = uv.x - i1;

                        j1 = (int)uv.y;
                        j2 = (j1+1)%N;
                        dv = uv.y - j1;

                        if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N && j1 >= 0 && j1 < N && j2 >= 0 && j2 < N) {
                                /* Bilinear interpolation */
                                v11 = V[N*j1 + i1]; /* [i1, j1] */
                                v12 = V[N*j2 + i1]; /* [i1, j2] */
                                v21 = V[N*j1 + i2]; /* [i2, j1] */
                                v22 = V[N*j2 + i2]; /* [i2, j2] */

                                Zreal = (1-du)*(1-dv)*v11.x + (1-du)*dv*v12.x + du*(1-dv)*v21.x + du*dv*v22.x;
                                Zimag = (1-du)*(1-dv)*v11.y + (1-du)*dv*v12.y + du*(1-dv)*v21.y + du*dv*v22.y;

                                Vm[i].x = Zreal;
                                Vm[i].y = Zimag;
                        }else{
                                weight[i] = 0.0f;
                        }
                }else{
                        //Vm[i].x = 0.0f;
                        //Vm[i].y = 0.0f;
                        weight[i] = 0.0f;
                }

        }

}


__global__ void vis_mod2(cufftComplex *Vm, cufftComplex *V, double3 *UVW, float *weight, double deltau, double deltav, long numVisibilities, long N)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        double f_j, f_k;
        int j, k;
        double2 uv;
        cufftComplex Z;

        if (i < numVisibilities) {

                uv.x = UVW[i].x/fabs(deltau);
                uv.y = UVW[i].y/fabs(deltav);

                f_j = round(uv.x + N/2);
                j = (int)f_j;
                f_j = f_j - j;

                f_k = round(uv.y + N/2);
                k = (int)f_k;
                f_k = f_k - k;


                if (j < N && k < N && j+1 < N && k+1 < N) {
                        /* Bilinear interpolation */
                        // Real part
                        Z.x = (1-f_j)*(1-f_k)*V[N*k+j].x + f_j*(1-f_k)*V[N*k+(j+1)].x + (1-f_j)*f_k*V[N*(k+1)+j].x + f_j*f_k*V[N*(k+1)+j+1].x;
                        // Imaginary part
                        Z.y = (1-f_j)*(1-f_k)*V[N*k+j].y + f_j*(1-f_k)*V[N*k+(j+1)].y + (1-f_j)*f_k*V[N*(k+1)+j].y + f_j*f_k*V[N*(k+1)+j+1].y;

                        Vm[i] = Z;
                }else{
                        weight[i] = 0.0f;
                }

        }

}


__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < numVisibilities) {
                Vr[i].x = Vm[i].x - Vo[i].x;
                Vr[i].y = Vm[i].y - Vo[i].y;
        }
}



__global__ void clipWNoise(cufftComplex *fg_image, float *noise, float *I, long N, float noise_cut, float MINPIX, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;


        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j] = 0.0;
                }
                else{
                        I[N*i+j] = -1.0 * eta * MINPIX;
                }

        }

        fg_image[N*i+j].x = I[N*i+j];
        fg_image[N*i+j].y = 0;
}

__global__ void clip2IWNoise(float *noise, float *I, long N, long M, float noise_cut, float MINPIX, float alpha_start, float eta, float threshold, int schedule)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j] = 0.0;
                }
                else{
                        I[N*i+j] = -1.0 * eta * MINPIX;
                }
                I[N*M+N*i+j] = 0.0f;
        }else{
                if(I[N*i+j] < threshold && schedule > 0) {
                        I[N*M+N*i+j] = 0.0f;
                }
        }
}


__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        gg[N*i+j] = g[N*i+j] * g[N*i+j];
        dgg[N*i+j] = (xi[N*i+j] + g[N*i+j]) * xi[N*i+j];
}

__global__ void getGGandDGG(float *gg, float *dgg, float* xi, float* g, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float gg_temp;
        float dgg_temp;

        gg_temp = g[M*N*image+N*i+j] * g[M*N*image+N*i+j];

        dgg_temp = (xi[M*N*image+N*i+j] + g[M*N*image+N*i+j]) * xi[M*N*image+N*i+j];

        gg[N*i+j] += gg_temp;
        dgg[N*i+j] += dgg_temp;
}

__global__ void clip(cufftComplex *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j].x < MINPIX && MINPIX >= 0.0) {
                I[N*i+j].x = MINPIX;
        }
        I[N*i+j].y = 0;
}

__global__ void clip(float *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j] < MINPIX && MINPIX >= 0.0) {
                I[N*i+j] = MINPIX;
        }
}

__global__ void clip2I(float *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j] < MINPIX && MINPIX >= 0.0) {
                I[N*i+j] = MINPIX;
        }
}

__global__ void newP(float *p, float *xi, float xmin, float MINPIX, float eta, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*i+j] *= xmin;
        if(p[N*i+j] + xi[N*i+j] > -1.0*eta*MINPIX) {
                p[N*i+j] += xi[N*i+j];
        }else{
                p[N*i+j] = -1.0*eta*MINPIX;
                xi[N*i+j] = 0.0;
        }
        //p[N*i+j].y = 0.0;
}

__global__ void newP(float*p, float*xi, float xmin, long N, long M, float MINPIX, float eta, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*M*image+N*i+j] *= xmin;

        if(p[N*M*image+N*i+j] + xi[N*M*image+N*i+j] > -1.0*eta*MINPIX) {
                p[N*M*image+N*i+j] += xi[N*M*image+N*i+j];
        }else{
                p[N*M*image+N*i+j] = -1.0*eta*MINPIX;
                xi[N*M*image+N*i+j] = 0.0;
        }
}

__global__ void newPNoPositivity(float *p, float *xi, float xmin, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*M*image+N*i+j] *= xmin;
        p[N*M*image+N*i+j] += xi[N*M*image+N*i+j];
}

__global__ void evaluateXt(float *xt, float *pcom, float *xicom, float x, float MINPIX, float eta, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(pcom[N*i+j] + x * xicom[N*i+j] > -1.0*eta*MINPIX) {
                xt[N*i+j] = pcom[N*i+j] + x * xicom[N*i+j];
        }else{
                xt[N*i+j] = -1.0*eta*MINPIX;
        }
        //xt[N*i+j].y = 0.0;
}

__global__ void evaluateXt(float*xt, float*pcom, float*xicom, float x, long N, long M, float MINPIX, float eta, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j] > -1.0*eta*MINPIX) {
                xt[N*M*image+N*i+j] = pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j];
        }else{
                xt[N*M*image+N*i+j] = -1.0*eta*MINPIX;
        }
}


__global__ void evaluateXtNoPositivity(float *xt, float *pcom, float *xicom, float x, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xt[N*M*image+N*i+j] = pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j];
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
        }

}

__device__ float calculateL1norm(float *I, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float c = I[N*M*index+N*i+j];

        float l1 = 0.0f;

        if(noise <= noise_cut) {
                l1 = normf(1, &c);
        }

        return l1;
}

__global__ void L1Vector(float *L1, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        L1[N*i+j] = calculateL1norm(I, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateS(float *I, float G, float eta, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float c = I[N*M*index+N*i+j];

        float S = 0.0f;

        if(noise <= noise_cut) {
                S = c * logf((c/G) + (eta + 1.0));
        }

        return S;
}
__global__ void SVector(float *S, float *noise, float *I, long N, long M, float noise_cut, float MINPIX, float eta, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        S[N*i+j] = calculateS(I, MINPIX, eta, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateQP(float *I, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float c, l, r, d, u;

        float qp = 0.0f;

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1))
                {
                        l = I[N*M*index+N*i+(j-1)];
                        r = I[N*M*index+N*i+(j+1)];
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];

                        qp = (c - l) * (c - l) +
                             (c - r) * (c - r) +
                             (c - u) * (c - u) +
                             (c - d) * (c - d);
                        qp /= 2.0;
                }else{
                        qp = c;
                }
        }

        return qp;
}
__global__ void QPVector(float *Q, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        Q[N*i+j] = calculateQP(I, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateTV(float *I, float noise, float noise_cut, int index, int M, int N)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float c, r, d;
        float tv = 0.0f;
        float dxy[2];

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                if(i < N-1 && j < N-1) {
                        r = I[N*M*index+N*i+(j+1)];
                        d = I[N*M*index+N*(i+1)+j];


                        dxy[0] = r - c;
                        dxy[1] = d - c;
                        tv = normf(2, dxy);
                }else{
                        tv = c;
                }
        }

        return tv;
}

__global__ void TVVector(float *TV, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        TV[N*i+j] = calculateTV(I, noise[N*i+j], noise_cut, index, M, N);

}

__device__ float calculateTSV(float *I, float noise, float noise_cut, int index, int M, int N)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float c, r, d;
        float tv = 0.0f;

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                if(i < N-1 && j < N-1) {
                        r = I[N*M*index+N*i+(j+1)];
                        d = I[N*M*index+N*(i+1)+j];

                        float dx = c - r;
                        float dy = c - d;
                        tv = dx * dx + dy * dy;
                }else{
                        tv = c;
                }
        }

        return tv;
}

__global__ void TSVVector(float *STV, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        STV[N*i+j] = calculateTSV(I, noise[N*i+j], noise_cut, index, M, N);

}

__device__ float calculateL(float *I, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float Dx, Dy;
        float L = 0.0f;
        float c, l, r, d, u;

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut)
        {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        l = I[N*M*index+N*i+(j-1)];
                        r = I[N*M*index+N*i+(j+1)];
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];

                        Dx = l - 2 * c + r;
                        Dy = u - 2 * c + d;
                        L = 0.5 * (Dx + Dy) * (Dx + Dy);
                }else{
                        L = c;
                }
        }

        return L;
}

__global__ void LVector(float *L, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        L[N*i+j] = calculateL(I, noise[N*i+j], noise_cut, index, M, N);
}

__global__ void searchDirection(float *g, float *xi, float *h, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j] = -xi[N*i+j];
        xi[N*i+j] = h[N*i+j] = g[N*i+j];
}

__global__ void searchDirection_LBFGS(float *xi, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[M*N*image+N*i+j] *= -1.0f;
}

__global__ void getDot_LBFGS_ff(float *aux_vector, float *vec_1, float *vec_2, int k, int h, int M, int N, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        aux_vector[N*i+j] = vec_1[M*N*image*k + M*N*image + (N*i+j)]*vec_2[M*N*image*h + M*N*image + (N*i+j)];
}

__global__ void updateQ (float *d_q, float alpha, float *d_y, int k, int M, int N, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        d_q[M*N*image+N*i+j] += alpha *d_y[M*N*image + M*N*k + (N*i+j)];
}

__global__ void getR (float *d_r, float *d_q, float scalar, int M, int N, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        d_r[M*N*image+N*i+j] = d_q[M*N*image+N*i+j] * scalar;
}

__global__ void calculateSandY (float *d_y, float *d_s, float *p, float *xi, float *p_old, float *xi_old, int iter, int M, int N, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        d_y[M*N*image*iter + M*N*image + (N*i+j)] = xi[M*N*image+N*i+j] - (-1.0f*xi_old[M*N*image+N*i+j]);
        d_s[M*N*image*iter + M*N*image + (N*i+j)] = p[M*N*image+N*i+j] - p_old[M*N*image+N*i+j];

}

__global__ void searchDirection(float* g, float* xi, float* h, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[M*N*image+N*i+j] = -xi[M*N*image+N*i+j];

        xi[M*N*image+N*i+j] = h[M*N*image+N*i+j] = g[M*N*image+N*i+j];
}

__global__ void newXi(float *g, float *xi, float *h, float gam, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j] = -xi[N*i+j];
        xi[N*i+j] = h[N*i+j] = g[N*i+j] + gam * h[N*i+j];
}

__global__ void newXi(float* g, float* xi, float* h, float gam, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[M*N*image+N*i+j] = -xi[M*N*image+N*i+j];

        xi[M*N*image+N*i+j] = h[M*N*image+N*i+j] = g[M*N*image+N*i+j] + gam * h[M*N*image+N*i+j];
}

__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*i+j] = 0.0;
        dChi2[N*i+j] = 0.0;
        dH[N*i+j] = 0.0;

}


__device__ float calculateDNormL1(float *I, float lambda, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float dL1 = 0.0f;

        float c = I[N*M*index+N*i+j];
        float sign = copysignf(1.0f, c);
        if(noise <= noise_cut)
                dL1 = sign;


        dL1 *= lambda;
        return dL1;
}

__global__ void DL1NormK(float *dL1, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dL1[N*i+j]  = calculateDNormL1(I, lambda, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateDS(float *I, float G, float eta, float lambda, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float dS = 0.0f;

        float c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                dS = logf((c / G) + (eta+1.0)) + 1.0/(1.0 + (((eta+1.0)*G) / c));
        }

        dS *= lambda;
        return dS;
}

__global__ void DS(float *dS, float *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dS[N*i+j]  = calculateDS(I, MINPIX, eta, lambda, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateDQ(float *I, float lambda, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float dQ = 0.0f;
        float c, d, u, r, l;

        c = I[N*M*index+N*i+j];

        if(noise <= noise_cut)
        {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];
                        r = I[N*M*index+N*i+(j+1)];
                        l = I[N*M*index+N*i+(j-1)];

                        dQ = 2 * (4 * c - d + u + r + l);
                }else{
                        dQ = c;
                }
        }

        dQ *= lambda;

        return dQ;
}

__global__ void DQ(float *dQ, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;


        dQ[N*i+j] = calculateDQ(I, lambda, noise[N*i+j], noise_cut, index, M, N);

}

__device__ float calculateDTV(float *I, float lambda, float noise, float noise_cut, int index, int M, int N)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float c, d, u, r, l, dl_corner, ru_corner;

        float num0, num1, num2;
        float den0[2], den1[2], den2[2];
        float norm0, norm1, norm2;
        float dtv = 0.0f;

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];
                        r = I[N*M*index+N*i+(j+1)];
                        l = I[N*M*index+N*i+(j-1)];
                        dl_corner = I[N*M*index+N*(i+1)+(j-1)];
                        ru_corner = I[N*M*index+N*(i-1)+(j+1)];

                        num0 = 2 * c - r - d;
                        num1 = c - l;
                        num2 = c - u;

                        den0[0] = c - r;
                        den0[1] = c - d;

                        den1[0] = l - c;
                        den1[1] = l - dl_corner;

                        den2[0] = u - ru_corner;
                        den2[1] = u - c;

                        norm0 = normf(2, den0);
                        norm1 = normf(2, den1);
                        norm2 = normf(2, den2);

                        if(norm0 == 0.0f || norm1 == 0.0f || norm2 == 0.0f) {
                                dtv = c;
                        }else{
                                dtv = num0/norm0 + num1/norm1 + num2/norm2;
                        }
                }else{
                        dtv = c;
                }
        }

        dtv *= lambda;

        return dtv;

}
__global__ void DTV(float *dTV, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float center, down, up, right, left, dl_corner, ru_corner;


        dTV[N*i+j] = calculateDTV(I, lambda, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateDTSV(float *I, float lambda, float noise, float noise_cut, int index, int M, int N)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float c, d, u, r, l, dl_corner, ru_corner;

        float dstv = 0.0f;

        c = I[N*M*index+N*i+j];
        if(noise <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];
                        r = I[N*M*index+N*i+(j+1)];
                        l = I[N*M*index+N*i+(j-1)];

                        dstv = 8.0f*c - 2.0f*(u + l + d + r);
                }else{
                        dstv = c;
                }
        }

        dstv *= lambda;

        return dstv;

}

__global__ void DTSV(float *dSTV, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        float center, down, up, right, left, dl_corner, ru_corner;


        dSTV[N*i+j] = calculateDTSV(I, lambda, noise[N*i+j], noise_cut, index, M, N);
}

__device__ float calculateDL(float *I, float lambda, float noise, float noise_cut, int index, int M, int N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float c, d, u, r, l, dl_corner, dr_corner, lu_corner, ru_corner, d2, u2, l2, r2;

        float dL = 0.0f;

        c = I[N*M*index+N*i+j];

        if(noise <= noise_cut)
        {
                if((i>1 && i<N-2) && (j>1 && j<N-2)) {
                        d = I[N*M*index+N*(i+1)+j];
                        u = I[N*M*index+N*(i-1)+j];
                        r = I[N*M*index+N*i+(j+1)];
                        l = I[N*M*index+N*i+(j-1)];
                        dl_corner = I[N*M*index+N*(i+1)+(j-1)];
                        dr_corner = I[N*M*index+N*(i+1)+(j+1)];
                        lu_corner = I[N*M*index+N*(i-1)+(j-1)];
                        ru_corner = I[N*M*index+N*(i-1)+(j+1)];
                        d2 = I[N*M*index+N*(i+2)+j];
                        u2 = I[N*M*index+N*(i-2)+j];
                        l2 = I[N*M*index+N*i+(j-2)];
                        r2 = I[N*M*index+N*i+(j+2)];

                        dL = 20 * c -
                             8 * (d - r - u - l) +
                             2 * (dl_corner + dr_corner + lu_corner + ru_corner) +
                             d2 + r2 + u2 + l2;
                }else
                        dL = 0.0f;

        }

        dL *= lambda;

        return dL;
}

__global__ void DL(float *dL, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dL[N*i+j] = calculateDL(I, lambda, noise[N*i+j], noise_cut, index, M, N);
}


__global__ void DChi2_SharedMemory(float *noise, float *dChi2, cufftComplex *Vr, double3 *UVW, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float ref_xobs, float ref_yobs, float phs_xobs, float phs_yobs, double DELTAX, double DELTAY, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, int primary_beam)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        cg::thread_block cta = cg::this_thread_block();

        extern __shared__ double s_array[];

        int x0 = phs_xobs;

        int y0 = phs_yobs;
        double x = (j - x0) * DELTAX * RPDEG_D;
        double y = (i - y0) * DELTAY * RPDEG_D;

        float Ukv, Vkv, cosk, sink, atten;

        double *u_shared = s_array;
        double *v_shared = (double*)&u_shared[numVisibilities];
        double *w_shared = (double*)&v_shared[numVisibilities];
        float *weight_shared = (float*)&w_shared[numVisibilities];
        cufftComplex *Vr_shared = (cufftComplex*)&weight_shared[numVisibilities];
        if(threadIdx.x == 0 && threadIdx.y == 0) {
                for(int v=0; v<numVisibilities; v++) {
                        u_shared[v] = UVW[v].x;
                        v_shared[v] = UVW[v].y;
                        w_shared[v] = w[v];
                        Vr_shared[v] = Vr[v];
                        printf("u: %f, v:%f, weight: %f, real: %f, imag: %f\n", u_shared[v], v_shared[v], w_shared[v], Vr_shared[v].x, Vr_shared[v].y);
                }
        }
        cg::sync(cta);


        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs, ref_yobs, DELTAX, DELTAY, primary_beam);

        float dchi2 = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * u_shared[v];
                        Vkv = y * v_shared[v];
        #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
        #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
        #endif
                        dchi2 += w_shared[v]*((Vr_shared[v].x * cosk) - (Vr_shared[v].y * sink));
                }

                dchi2 *= fg_scale * atten;
                dChi2[N*i+j] = dchi2;
        }
}


__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, double3 *UVW, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float ref_xobs, float ref_yobs, float phs_xobs, float phs_yobs, double DELTAX, double DELTAY, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, int primary_beam)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        int x0 = phs_xobs;
        int y0 = phs_yobs;
        double x = (j - x0) * DELTAX * RPDEG_D;
        double y = (i - y0) * DELTAY * RPDEG_D;
        //double z = sqrt(1-x*x-y*y)-1;

        float Ukv, Vkv, Wkv, cosk, sink, atten;

        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, ref_xobs, ref_yobs, DELTAX, DELTAY, primary_beam);

        float dchi2 = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * UVW[v].x;
                        Vkv = y * UVW[v].y;
                        //Wkv = z * UVW[v].z;
                        #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
                        #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
                        #endif
                        dchi2 += w[v]*((Vr[v].x * cosk) - (Vr[v].y * sink));
                }

                dchi2 *= fg_scale * atten;
                dChi2[N*i+j] = dchi2;
        }
}


__global__ void DPhi(float *dphi, float *dchi2, float *dH, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*i+j] = dchi2[N*i+j] + dH[N*i+j];
}

__global__ void AddToDPhi(float *dphi, float *dgi, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*M*index+N*i+j] += dgi[N*i+j];
}

__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        x[N*i+j] = xc[N*i+j].x - lambda*gc[N*i+j];
}

__global__ void projection(float *px, float *x, float MINPIX, long N){

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;


        if(INFINITY < x[N*i+j]) {
                px[N*i+j] = INFINITY;
        }else{
                px[N*i+j] = x[N*i+j];
        }

        if(MINPIX > px[N*i+j]) {
                px[N*i+j] = MINPIX;
        }else{
                px[N*i+j] = px[N*i+j];
        }
}

__global__ void normVectorCalculation(float *normVector, float *gc, long N){
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        normVector[N*i+j] = gc[N*i+j] * gc[N*i+j];
}

__global__ void copyImage(cufftComplex *p, float *device_xt, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        p[N*i+j].x = device_xt[N*i+j];
}

__global__ void calculateInu(cufftComplex *I_nu, float* I, float nu, float nu_0, float MINPIX, float eta, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

        nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[M*N+N*i+j] * 100.0f;

        nudiv_pow_alpha = powf(nudiv, alpha);

        I_nu[N*i+j].x = I_nu_0 * nudiv_pow_alpha;

        if(I_nu[N*i+j].x < -1.0*eta*MINPIX) {
                I_nu[N*i+j].x = -1.0*eta*MINPIX;
        }

        I_nu[N*i+j].y = 0.0f;
}

__global__ void DChi2_total_alpha(float *noise, float *dchi2_total, float *dchi2, float *I, float nu, float nu_0, float noise_cut, float fg_scale, float threshold, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dalpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j] * 100.0f;

        dI_nu_0 = powf(nudiv, alpha);
        dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

        if(noise[N*i+j] <= noise_cut) {
                if(I_nu_0 > threshold) {
                        dchi2_total[N*M+N*i+j] += dchi2[N*i+j] * dalpha;
                }
                else{
                        dchi2_total[N*M+N*i+j] += 0.0f;
                }
        }
}

__global__ void DChi2_total_I_nu_0(float *noise, float *dchi2_total, float *dchi2, float *I, float nu, float nu_0, float noise_cut, float fg_scale, float threshold, long N, long M)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j] * 100.0f;

        dI_nu_0 = powf(nudiv, alpha);
        //dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

        if(noise[N*i+j] <= noise_cut)
                dchi2_total[N*i+j] += dchi2[N*i+j] * dI_nu_0;

}

__global__ void chainRule2I(float *chain, float *noise, float *I, float nu, float nu_0, float noise_cut, float fg_scale, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dalpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j];

        dI_nu_0 = powf(nudiv, alpha);
        dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

        chain[N*i+j] = dI_nu_0;
        chain[N*M+N*i+j] = dalpha;
}

__global__ void DChi2_2I(float *noise, float *chain, float *I, float *dchi2, float *dchi2_total, float threshold, float noise_cut, int image, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] <= noise_cut && image)
        {
                if(I[N*i+j] > threshold) {
                        dchi2_total[N*i+j] += dchi2[N*i+j] * chain[N*M+N*i+j];
                }else{
                        dchi2_total[N*i+j] += 0.0f;
                }

        }else if(noise[N*i+j] <= noise_cut) {
                dchi2_total[N*i+j] += dchi2[N*i+j] * chain[N*i+j];
        }
}


__global__ void I_nu_0_Noise(float *noise_I, float *images, float *noise, float noise_cut, float nu, float nu_0, float *w, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, double DELTAX, double DELTAY, float sum_weights, long N, long M, int primary_beam)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float alpha, nudiv, nudiv_pow_alpha, atten;

        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs, DELTAX, DELTAY, primary_beam);

        nudiv = nu/nu_0;
        alpha = images[N*M+N*i+j];
        nudiv_pow_alpha = powf(nudiv, 2.0f*alpha);

        if(noise[N*i+j] <= noise_cut) {
                noise_I[N*i+j] += atten * atten * sum_weights * nudiv_pow_alpha;
        }else{
                noise_I[N*i+j] = 0.0f;
        }



}


__global__ void alpha_Noise(float *noise_I, float *images, float nu, float nu_0, float *w, double3 *UVW, cufftComplex *Vr, float *noise, float noise_cut, double DELTAX, double DELTAY, float xobs, float yobs, float antenna_diameter, float pb_factor, float pb_cutoff, float fg_scale, long numVisibilities, long N, long M, int primary_beam)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, Ukv, Vkv, cosk, sink, x, y, dchi2, sum_noise, atten;
        int x0, y0;

        x0 = xobs;
        y0 = yobs;
        x = (j - x0) * DELTAX * RPDEG_D;
        y = (i - y0) * DELTAY * RPDEG_D;

        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs, DELTAX, DELTAY, primary_beam);

        nudiv = nu/nu_0;
        I_nu_0 = images[N*i+j];
        alpha = images[N*M+N*i+j];
        nudiv_pow_alpha = powf(nudiv, alpha);

        I_nu = I_nu_0 * nudiv_pow_alpha;
        log_nu = logf(nudiv);

        sum_noise = 0.0f;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * UVW[v].x;
                        Vkv = y * UVW[v].y;
      #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
      #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
      #endif
                        dchi2 = ((Vr[v].x * cosk) - (Vr[v].y * sink));
                        sum_noise += w[v] * (atten * I_nu * fg_scale + dchi2);
                }
                if(sum_noise <= 0.0f)
                        noise_I[N*M+N*i+j]+= 0.0f;
                else
                        noise_I[N*M+N*i+j] += log_nu * log_nu * atten * I_nu * fg_scale * sum_noise;
        }else{
                noise_I[N*M+N*i+j] = 0.0f;
        }
}

__global__ void noise_reduction(float *noise_I, long N, long M){
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise_I[N*i+j] > 0.0f)
                noise_I[N*i+j] = 1/sqrt(noise_I[N*i+j]);
        else
                noise_I[N*i+j] = 0.0f;

        if(noise_I[N*M+N*i+j] > 0.0f)
                noise_I[N*M+N*i+j] = 1/sqrt(noise_I[N*M+N*i+j]);
        else
                noise_I[N*M+N*i+j] = 0.0f;
}


__host__ float chi2(float *I, VirtualImageProcessor *ip)
{

        cudaSetDevice(firstgpu);

        float resultPhi = 0.0;
        float resultchi2  = 0.0;
        float resultS  = 0.0;

        if(clip_flag) {
                ip->clip(I);
        }

        ip->clipWNoise(I);

        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        if(num_gpus == 1) {
                                cudaSetDevice(selected);
                                for(int i=0; i<datasets[d].data.total_frequencies; i++) {

                                        ip->calculateInu(vars_gpu[0].device_I_nu, I, datasets[d].fields[f].nu[i]);

                                        ip->apply_beam(vars_gpu[0].device_I_nu, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, datasets[d].fields[f].nu[i], datasets[d].antennas[0].primary_beam);

                                        //FFT 2D
                                        FFT2D(vars_gpu[0].device_V, vars_gpu[0].device_I_nu, vars_gpu[0].plan, M, N, false);

                                        // PHASE_ROTATE
                                        phase_rotate <<< numBlocksNN, threadsPerBlockNN >>>
                                        (vars_gpu[0].device_V, M, N, datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        for(int s=0; s<datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {

                                                        gpuErrchk(cudaMemset(vars_gpu[0].device_chi2, 0, sizeof(float) * max_number_vis));

                                                        // BILINEAR INTERPOLATION
                                                        vis_mod <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                                        (datasets[d].fields[f].device_visibilities[i][s].Vm, vars_gpu[0].device_V, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].weight, deltau, deltav, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], N);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                        //DFT 2D
                                                        /*DFT2D <<<fields[f].visibilities[i][s].numBlocksUV,
                                                           fields[f].visibilities[i][s].threadsPerBlockUV>>>(fields[f].device_visibilities[i][s].Vm, vars_gpu[0].device_I_nu, fields[f].device_visibilities[i][s].uvw, device_noise_image, noise_cut, fields[f].phs_xobs, fields[f].phs_yobs, DELTAX, DELTAY, M, N, fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                           gpuErrchk(cudaDeviceSynchronize());*/

                                                        // RESIDUAL CALCULATION
                                                        residual <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                                        (datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].Vm, datasets[d].fields[f].device_visibilities[i][s].Vo, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                        ////chi 2 VECTOR
                                                        chi2Vector <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                                        (vars_gpu[0].device_chi2, datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                        //REDUCTIONS
                                                        //chi2
                                                        resultchi2 += deviceReduce<float>(vars_gpu[0].device_chi2,
                                                                                          datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                }
                                        }
                                }
                        }else{
                              #pragma omp parallel for schedule(static,1)
                                for (int i = 0; i < datasets[d].data.total_frequencies; i++)
                                {
                                        float result = 0.0;
                                        unsigned int j = omp_get_thread_num();
                                        //unsigned int num_cpu_threads = omp_get_num_threads();
                                        // set and check the CUDA device for this CPU thread
                                        int gpu_id = -1;
                                        cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                        cudaGetDevice(&gpu_id);
                                        ip->calculateInu(vars_gpu[i % num_gpus].device_I_nu, I, datasets[d].fields[f].nu[i]);

                                        ip->apply_beam(vars_gpu[i % num_gpus].device_I_nu, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].ref_xobs,
                                                       datasets[d].fields[f].ref_yobs, datasets[d].fields[f].nu[i], datasets[d].antennas[0].primary_beam);

                                        //FFT 2D
                                        FFT2D(vars_gpu[i % num_gpus].device_V, vars_gpu[i % num_gpus].device_I_nu, vars_gpu[i % num_gpus].plan, M, N, false);

                                        //PHASE_ROTATE
                                        phase_rotate <<< numBlocksNN, threadsPerBlockNN >>> (vars_gpu[i % num_gpus].device_V, M, N, datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());
                                        for(int s=0; s<datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {

                                                        gpuErrchk(cudaMemset(vars_gpu[i % num_gpus].device_chi2, 0, sizeof(float)*max_number_vis));
                                                        // BILINEAR INTERPOLATION
                                                        vis_mod <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                                        (datasets[d].fields[f].device_visibilities[i][s].Vm, vars_gpu[i % num_gpus].device_V, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].weight, deltau, deltav, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], N);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                        // RESIDUAL CALCULATION
                                                        residual <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>>
                                                        (datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].Vm, datasets[d].fields[f].device_visibilities[i][s].Vo, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                        ////chi2 VECTOR
                                                        chi2Vector <<< datasets[d].fields[f].device_visibilities[i][s].numBlocksUV,
                                                                datasets[d].fields[f].device_visibilities[i][s].threadsPerBlockUV >>> (vars_gpu[i % num_gpus].device_chi2, datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        gpuErrchk(cudaDeviceSynchronize());


                                                        result = deviceReduce<float>(vars_gpu[i % num_gpus].device_chi2,
                                                                                     datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        //REDUCTIONS
                                                        //chi2
                                              #pragma omp atomic
                                                        resultchi2 += result;
                                                }
                                        }
                                }
                        }
                }
        }


        cudaSetDevice(firstgpu);

        resultPhi = (0.5 * resultchi2);

        final_chi2 = resultchi2;

        return resultPhi;
};

__host__ void dchi2(float *I, float *dxi2, float *result_dchi2, VirtualImageProcessor *ip)
{

        cudaSetDevice(firstgpu);

        if(clip_flag) {
                ip->clip(I);
        }

        for(int d=0; d<nMeasurementSets; d++) {
                if(num_gpus == 1) {
                        for(int f=0; f<datasets[d].data.nfields; f++) {
                                for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                        for(int s=0; s < datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
                                                        gpuErrchk(cudaMemset(vars_gpu[0].device_dchi2, 0, sizeof(float) *
                                                                             M*N));
                                                        //size_t shared_memory;
                                                        //shared_memory = 3*fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(double) + fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(float) + fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex);
                                                        DChi2 <<< numBlocksNN, threadsPerBlockNN >>>
                                                        (device_noise_image, vars_gpu[0].device_dchi2, datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].weight, N, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], fg_scale, noise_cut, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs, DELTAX, DELTAY, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].nu[i], datasets[d].antennas[0].primary_beam);
                                                        //DChi2_SharedMemory<<<numBlocksNN, threadsPerBlockNN, shared_memory>>>(device_noise_image, vars_gpu[0].device_dchi2, fields[f].device_visibilities[i][s].Vr, fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].weight, N, fields[f].numVisibilitiesPerFreqPerStoke[i][s], fg_scale, noise_cut, fields[f].ref_xobs, fields[f].ref_yobs, fields[f].phs_xobs, fields[f].phs_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].nu[i]);
                                                        gpuErrchk(cudaDeviceSynchronize());


                                                        if (flag_opt % 2 == 0)
                                                                DChi2_total_I_nu_0 <<< numBlocksNN, threadsPerBlockNN >>>
                                                                (device_noise_image, result_dchi2, vars_gpu[0].device_dchi2, I, datasets[d].fields[f].nu[i], nu_0, noise_cut, fg_scale, threshold, N, M);
                                                        else
                                                                DChi2_total_alpha <<< numBlocksNN, threadsPerBlockNN >>>
                                                                (device_noise_image, result_dchi2, vars_gpu[0].device_dchi2, I, datasets[d].fields[f].nu[i], nu_0, noise_cut, fg_scale, threshold, N, M);
                                                        gpuErrchk(cudaDeviceSynchronize());

                                                }
                                        }
                                }
                        }
                }else{
                        for(int f=0; f<datasets[d].data.nfields; f++) {
                          #pragma omp parallel for schedule(static,1)
                                for (int i = 0; i < datasets[d].data.total_frequencies; i++)
                                {
                                        unsigned int j = omp_get_thread_num();
                                        //unsigned int num_cpu_threads = omp_get_num_threads();
                                        // set and check the CUDA device for this CPU thread
                                        int gpu_id = -1;
                                        cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                        cudaGetDevice(&gpu_id);
                                        for(int s=0; s<datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
                                                        gpuErrchk(cudaMemset(vars_gpu[i % num_gpus].device_dchi2, 0, sizeof(float) *
                                                                             M*N));
                                                        //size_t shared_memory;
                                                        //shared_memory = 3*fields[f].numVisibilitiesPerFreq[i]*sizeof(float) + fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex);
                                                        DChi2 <<< numBlocksNN, threadsPerBlockNN >>> (device_noise_image, vars_gpu[i % num_gpus].device_dchi2, datasets[d].fields[f].device_visibilities[i][s].Vr, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].weight, N, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], fg_scale, noise_cut, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs, DELTAX, DELTAY, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].nu[i], datasets[d].antennas[0].primary_beam);
                                                        //DChi2<<<numBlocksNN, threadsPerBlockNN, shared_memory>>>(device_noise_image, vars_gpu[i%num_gpus].device_dchi2, fields[f].device_visibilities[i][s].Vr, fields[f].device_visibilities[i][s].uvw, fields[f].device_visibilities[i][s].weight, N, fields[f].numVisibilitiesPerFreqPerStoke[i][s], fg_scale, noise_cut, fields[f].ref_xobs, fields[f].ref_yobs, fields[f].phs_xobs, fields[f].phs_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].nu[i]);
                                                        gpuErrchk(cudaDeviceSynchronize());


                                          #pragma omp critical
                                                        {
                                                                if (flag_opt % 2 == 0)
                                                                        DChi2_total_I_nu_0 <<< numBlocksNN, threadsPerBlockNN >>>
                                                                        (device_noise_image, result_dchi2, vars_gpu[i % num_gpus].device_dchi2, I, datasets[d].fields[f].nu[i], nu_0, noise_cut, fg_scale, threshold, N, M);
                                                                else
                                                                        DChi2_total_alpha <<< numBlocksNN, threadsPerBlockNN >>>
                                                                        (device_noise_image, result_dchi2, vars_gpu[i % num_gpus].device_dchi2, I, datasets[d].fields[f].nu[i], nu_0, noise_cut, fg_scale, threshold, N, M);
                                                                gpuErrchk(cudaDeviceSynchronize());
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }


        cudaSetDevice(firstgpu);

};

__host__ void linkAddToDPhi(float *dphi, float *dgi, int index)
{
        cudaSetDevice(firstgpu);
        AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void defaultNewP(float*p, float*xi, float xmin, int image)
{
        newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M, image);
};

__host__ void particularNewP(float*p, float*xi, float xmin, int image)
{
        newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M, initial_values[image], eta, image);
};

__host__ void defaultEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image)
{
        evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x, N, M, image);
};

__host__ void particularEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image)
{
        evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x, N, M, initial_values[image], eta, image);
};

__host__ void linkClipWNoise2I(float *I)
{
        clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, I, N, M, noise_cut, initial_values[0], initial_values[1], eta, threshold, flag_opt);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkApplyBeam2I(cufftComplex *image, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float freq, int primary_beam)
{
        apply_beam2I<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, image, N, xobs, yobs, fg_scale, freq, DELTAX, DELTAY, primary_beam);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkCalculateInu2I(cufftComplex *image, float *I, float freq)
{
        calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(image, I, freq, nu_0, initial_values[0], eta, N, M);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkClip(float *I)
{
        clip<<<numBlocksNN, threadsPerBlockNN>>>(I, N, initial_values[0]);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkChain2I(float *chain, float freq, float *I)
{
        chainRule2I<<<numBlocksNN, threadsPerBlockNN>>>(chain, device_noise_image, I, freq, nu_0, noise_cut, fg_scale, N, M);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ float L1Norm(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        float resultL1norm = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                L1Vector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultL1norm  = deviceReduce<float>(ds, M*N);
        }

        return resultL1norm;
};

__host__ void DL1Norm(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DL1NormK<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};


__host__ float SEntropy(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        float resultS = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                SVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, initial_values[index], eta, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DEntropy(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DS<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, initial_values[index], eta, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};

__host__ float laplacian(float *I, float * ds, float penalization_factor, int mod, int order, int imageIndex)
{
        cudaSetDevice(firstgpu);

        float resultS = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                LVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, imageIndex);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        return resultS;
};

__host__ void DLaplacian(float *I, float *dgi, float penalization_factor, float mod, float order, float index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DL<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};

__host__ float quadraticP(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        float resultS = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                QPVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DQuadraticP(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DQ<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};

__host__ float totalvariation(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        float resultS = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                TVVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DTVariation(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DTV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};

__host__ float TotalSquaredVariation(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        float resultS = 0.0f;
        if(iter > 0 && penalization_factor)
        {
                TSVVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DTSVariation(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        cudaSetDevice(firstgpu);

        if(iter > 0 && penalization_factor)
        {
                if(flag_opt%2 == index) {
                        DTSV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }
};

__host__ void calculateErrors(Image *image){

        float *errors = image->getErrorImage();


        cudaSetDevice(firstgpu);

        gpuErrchk(cudaMalloc((void**)&errors, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemset(errors, 0, sizeof(float)*M*N*image->getImageCount()));
        float sum_weights;
        for(int d=0; d<nMeasurementSets; d++) {
                if(num_gpus == 1) {
                        cudaSetDevice(selected);
                        for(int f=0; f<datasets[d].data.nfields; f++) {
                                for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                        for(int s=0; s<datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreq[i] > 0) {
                                                        sum_weights = deviceReduce<float>(datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                                        I_nu_0_Noise <<< numBlocksNN, threadsPerBlockNN >>>
                                                        (errors, image->getImage(), device_noise_image, noise_cut, datasets[d].fields[f].nu[i], nu_0, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, DELTAX, DELTAY, sum_weights, N, M, datasets[d].antennas[0].primary_beam);
                                                        gpuErrchk(cudaDeviceSynchronize());
                                                        alpha_Noise <<< numBlocksNN, threadsPerBlockNN >>>
                                                        (errors, image->getImage(), datasets[d].fields[f].nu[i], nu_0, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].Vr, device_noise_image, noise_cut, DELTAX, DELTAY, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, fg_scale, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], N, M, datasets[d].antennas[0].primary_beam);
                                                        gpuErrchk(cudaDeviceSynchronize());
                                                }
                                        }
                                }
                        }
                }else{
                        for(int f=0; f<datasets[d].data.nfields; f++) {
                          #pragma omp parallel for private(sum_weights) schedule(static,1)
                                for (int i = 0; i < datasets[d].data.total_frequencies; i++)
                                {
                                        unsigned int j = omp_get_thread_num();
                                        //unsigned int num_cpu_threads = omp_get_num_threads();
                                        // set and check the CUDA device for this CPU thread
                                        int gpu_id = -1;
                                        cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                        cudaGetDevice(&gpu_id);
                                        for(int s=0; s<datasets[d].data.nstokes; s++) {
                                                if (datasets[d].fields[f].numVisibilitiesPerFreq[i] > 0) {
                                                        sum_weights = deviceReduce<float>(datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                          #pragma omp critical
                                                        {
                                                                I_nu_0_Noise << < numBlocksNN, threadsPerBlockNN >> >
                                                                (errors, image->getImage(), device_noise_image, noise_cut, datasets[d].fields[f].nu[i], nu_0, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, DELTAX, DELTAY, sum_weights, N, M, datasets[d].antennas[0].primary_beam);
                                                                gpuErrchk(cudaDeviceSynchronize());
                                                                alpha_Noise << < numBlocksNN, threadsPerBlockNN >> >
                                                                (errors, image->getImage(), datasets[d].fields[f].nu[i], nu_0, datasets[d].fields[f].device_visibilities[i][s].weight, datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].Vr, device_noise_image, noise_cut, DELTAX, DELTAY, datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, datasets[d].antennas[0].antenna_diameter, datasets[d].antennas[0].pb_factor, datasets[d].antennas[0].pb_cutoff, fg_scale, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s], N, M, datasets[d].antennas[0].primary_beam);
                                                                gpuErrchk(cudaDeviceSynchronize());
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(errors, N, M);
        gpuErrchk(cudaDeviceSynchronize());

        image->setErrorImage(errors);

}
