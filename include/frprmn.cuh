#ifndef FRPRMN_CUH
#define FRPRMN_CUH
#include "linmin.cuh"

__host__ void frprmn(float *p, float ftol, float *fret, float (*func)(float*), void (*dfunc)(float*, float*));

class ConjugateGradient : public Optimizator
{
public:
__host__ void allocateMemoryGpu();
__host__ void deallocateMemoryGpu();
__host__ void optimize();
private:
float ftol = 0;
float fret = 0;
float gg, dgg, gam, fp;
float *device_g, *device_h, *xi;
float *device_gg_vector, *device_dgg_vector;
int configured = 1;
};

#endif
