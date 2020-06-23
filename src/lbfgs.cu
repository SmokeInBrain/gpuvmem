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

#include "lbfgs.cuh"

extern long M;
extern long N;
extern int iter;

extern ObjectiveFunction *testof;
extern Image *I;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int threadsVectorNN;
extern int blocksVectorNN;

extern float MINPIX, ftol;
extern int verbose_flag;
extern int flag_opt;

#define EPS 1.0e-10
extern int it_maximum;

#define ALPHA 1.e-4

#define FREEALL cudaFree(d_y); cudaFree(d_s); cudaFree(xi); cudaFree(xi_old); cudaFree(p_old); cudaFree(norm_vector);

__host__ void LBFGS::allocateMemoryGpu()
{
        gpuErrchk(cudaMalloc((void**)&d_y, sizeof(float)*M*N*K*image->getImageCount()));
        gpuErrchk(cudaMemset(d_y, 0, sizeof(float)*M*N*K*image->getImageCount()));

        gpuErrchk(cudaMalloc((void**)&d_s, sizeof(float)*M*N*K*image->getImageCount()));
        gpuErrchk(cudaMemset(d_s, 0, sizeof(float)*M*N*K*image->getImageCount()));

        gpuErrchk(cudaMalloc((void**)&p_old, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemset(p_old, 0, sizeof(float)*M*N*image->getImageCount()));

        gpuErrchk(cudaMalloc((void**)&xi, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemset(xi, 0, sizeof(float)*M*N*image->getImageCount()));

        gpuErrchk(cudaMalloc((void**)&xi_old, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemset(xi_old, 0, sizeof(float)*M*N*image->getImageCount()));

        gpuErrchk(cudaMalloc((void**)&norm_vector, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(norm_vector, 0, sizeof(float)*M*N));
};
__host__ void LBFGS::deallocateMemoryGpu()
{
        FREEALL
};

__host__ void LBFGS::optimize()
{
        printf("\n\nStarting Lbfgs\n\n");
        double start, end;
        I = image;
        flag_opt = this->flag;
        allocateMemoryGpu();
        testof = of;
        if(configured) {
                of->configure(N, M, image->getImageCount());
                configured = 0;
        }

        fp = of->calcFunction(image->getImage());
        if(verbose_flag) {
                printf("Starting function value = %f\n", fp);
        }
        of->calcGradient(image->getImage(),xi);


        //gpuErrchk(cudaMemcpy(p_old, image->getImage(), sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
        //gpuErrchk(cudaMemcpy(xi_old, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));

        for(int i=0; i < image->getImageCount(); i++)
        {
                searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(xi, M, N, i); //Search direction
                gpuErrchk(cudaDeviceSynchronize());
        }

        ////////////////////////////////////////////////////////////////
        for(int i=1; i <= it_maximum; i++) {
                start = omp_get_wtime();
                iter = i;
                norm = 0.0f;
                if(verbose_flag) {
                        printf("\n\n********** Iteration %d **********\n\n", i);
                }

                gpuErrchk(cudaMemcpy(p_old, image->getImage(), sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
                gpuErrchk(cudaMemcpy(xi_old, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));

                linmin(image->getImage(), xi, &fret, NULL);

                if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS)) {
                        printf("Exit due to tolerance\n");
                        of->calcFunction(I->getImage());
                        deallocateMemoryGpu();
                        return;
                }

                for(int i=0; i < image->getImageCount(); i++)
                {
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(norm_vector, xi, xi, 0, 0, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                        norm += deviceReduce<float>(norm_vector, M*N);
                }

                if(norm <= ftol) {
                        printf("Exit due to norm = 0\n");
                        of->calcFunction(image->getImage());
                        deallocateMemoryGpu();
                        return;
                }


                fp= of->calcFunction(image->getImage());
                if(verbose_flag) {
                        printf("Function value = %f\n", fp);
                }
                of->calcGradient(image->getImage(),xi);

                for(int i=0; i < image->getImageCount(); i++)
                {
                        calculateSandY<<<numBlocksNN, threadsPerBlockNN>>>(d_y, d_s, image->getImage(), xi, p_old, xi_old, (iter-1)%K, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                }

                LBFGS_recursion(d_y, d_s, xi, std::min(K,iter), (iter-1)%K, M, N);

                end = omp_get_wtime();
                double wall_time = end-start;
                if(verbose_flag) {
                        printf("Time: %lf seconds\n", i, wall_time);
                }
        }
        printf("Too many iterations in LBFGS\n");
        of->calcFunction(image->getImage());
        deallocateMemoryGpu();
        return;
};

__host__ void LBFGS::LBFGS_recursion(float *d_y, float *d_s, float *xi, int par_M, int lbfgs_it, int M, int N){
        float **alpha, *aux_vector;
        float *d_r, *d_q;
        float rho = 0.0f;
        float rho_den;
        float beta = 0.0f;
        float sy = 0.0f;
        float yy = 0.0f;
        float sy_yy = 0.0f;
        alpha = (float**)malloc(image->getImageCount()*sizeof(float*));
        for(int i=0; i<image->getImageCount(); i++) {
                alpha[i] = (float*)malloc(par_M*sizeof(float));
        }

        for(int i=0; i<image->getImageCount(); i++) {
                memset (alpha[i],0,par_M*sizeof(float));
        }


        gpuErrchk(cudaMalloc((void**)&aux_vector, sizeof(float)*M*N));
        gpuErrchk(cudaMalloc((void**)&d_q, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMalloc((void**)&d_r, sizeof(float)*M*N*image->getImageCount()));

        gpuErrchk(cudaMemset(aux_vector, 0, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(d_r, 0, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemcpy(d_q, xi, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));



        for(int i=0; i < I->getImageCount(); i++)
        {

                for(int k = par_M - 1; k >= 0; k--) {
                        //Rho_k = 1.0/(y_k's_k);
                        //printf("Image %d, Iter :%d\n", i, k);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                        rho_den = deviceReduce<float>(aux_vector, M*N);
                        if(rho_den != 0.0f)
                                rho = 1.0/rho_den;
                        else
                                rho = 0.0f;
                        //printf("1. rho %f\n", rho);
                        //alpha_k = Rho_k x (s_k' * q);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_s, d_q, k, 0, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                        alpha[i][k] = rho * deviceReduce<float>(aux_vector, M*N);
                        //printf("1. alpha %f\n", alpha[i][k]);
                        //q = q - alpha_k * y_k;
                        updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, -alpha[i][k], d_y, k, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }

        //y_0'y_0
        //(s_0'y_0)/(y_0'y_0)
        for(int i=0; i < I->getImageCount(); i++)
        {
                getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, lbfgs_it, lbfgs_it, M, N, i);
                gpuErrchk(cudaDeviceSynchronize());
                sy = deviceReduce<float>(aux_vector, M*N);

                getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_y, lbfgs_it, lbfgs_it, M, N, i);
                gpuErrchk(cudaDeviceSynchronize());
                yy = deviceReduce<float>(aux_vector, M*N);

                if(yy!=0.0)
                        sy_yy += sy/yy;
                else
                        sy_yy += 0.0f;

                //printf("sy: %f, yy: %f, sy_yy: %f\n",sy, yy, sy_yy);
        }




        for(int i=0; i < I->getImageCount(); i++)
        {
                // r = q x ((s_0'y_0)/(y_0'y_0));
                getR<<<numBlocksNN, threadsPerBlockNN>>>(d_r, d_q, sy_yy, M, N, i);
                gpuErrchk(cudaDeviceSynchronize());

                for (int k = 0; k < par_M; k++) {
                        //Rho_k = 1.0/(y_k's_k);
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                        //Calculate rho
                        rho_den = deviceReduce<float>(aux_vector, M*N);
                        if(rho_den != 0.0f)
                                rho = 1.0/rho_den;
                        else
                                rho = 0.0f;
                        //beta = rho * y_k' * r;
                        getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_r, k, 0, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                        beta = rho * deviceReduce<float>(aux_vector, M*N);
                        //printf("2. image %d - iter %d - rho: %f, alpha: %f, beta: %f, alpha-beta: %f\n", i, k, rho, alpha[i][k], beta, alpha[i][k]-beta);
                        //r = r + s_k * (alpha_k - beta)
                        updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_r, alpha[i][k]-beta, d_s, k, M, N, i);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }

        //Changing the sign to -d_r
        for(int i=0; i < image->getImageCount(); i++)
        {
                searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(d_r, M, N, i); //Search direction
                gpuErrchk(cudaDeviceSynchronize());
        }

        gpuErrchk(cudaMemcpy(xi, d_r, sizeof(float)*M*N*image->getImageCount(), cudaMemcpyDeviceToDevice));
        cudaFree(aux_vector);
        cudaFree(d_q);
        cudaFree(d_r);
        for(int i=0; i<image->getImageCount(); i++) {
                free(alpha[i]);
        }
        free(alpha);
};

namespace {
Optimizator *CreateLbfgs()
{
        return new LBFGS;
};
const int LbfgsId = 1;
const bool RegisteredLbgs = Singleton<OptimizatorFactory>::Instance().RegisterOptimizator(LbfgsId, CreateLbfgs);
};
