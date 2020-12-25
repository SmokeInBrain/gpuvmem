#ifndef GL1NORM_CUH
#define GL1NORM_CUH

#include "framework.cuh"
#include "functions.cuh"


class GL1Norm : public Fi
{
private:
float *prior;
float normalization_factor;
public:
GL1Norm();
GL1Norm(float *prior);
GL1Norm(float *prior, float normalization_factor);
GL1Norm(std::vector<float> prior);
GL1Norm(std::vector<float> prior, float normalization_factor);
~GL1Norm();
float getNormalizationFactor();
void setNormalizationFactor(float normalization_factor);
void setPrior(float *prior);
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
void normalizePrior();
};

#endif
