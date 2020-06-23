#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <math.h>
#include <string>
#include <functional>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include "MSFITSIO.cuh"
#include "copyrightwarranty.cuh"
#include <cooperative_groups.h>

typedef struct varsPerGPU {
        float *device_chi2;
        float *device_dchi2;
        cufftHandle plan;
        cufftComplex *device_I_nu;
        cufftComplex *device_V;
}varsPerGPU;

typedef struct variables {
        char *input;
        char *output;
        char *inputdat;
        char *modin;
        char *ofile;
        char *path;
        char *output_image;
        char *gpus;
        char *initial_values;
        char *penalization_factors;
        int blockSizeX;
        int blockSizeY;
        int blockSizeV;
        int it_max;
        int gridding;
        float noise;
        float noise_cut;
        float randoms;
        float eta;
        float nu_0;
        float robust_param;
        float threshold;
} Vars;

typedef struct functionMap
{
        void (*newP)(float*, float*, float, int);
        void (*evaluateXt)(float*, float*, float*, float, int);
} imageMap;

class VirtualImageProcessor
{
public:
virtual void clip(float *I) = 0;
virtual void clipWNoise(float *I) = 0;
virtual void apply_beam(cufftComplex *image, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float freq, int primary_beam) = 0;
virtual void calculateInu(cufftComplex *image, float *I, float freq) = 0;
virtual void chainRule(float *I, float freq) = 0;
virtual void configure(int I) = 0;
protected:
float *chain;
cufftComplex *fg_image;
int image_count;
};


template<class T>
class Singleton
{
public:
static T& Instance()
{
        static T instance;
        return instance;
}
private:
Singleton(){
};
Singleton(T const&)         = delete;
void operator=(T const&) = delete;
};

class Fi
{
public:
virtual float calcFi(float *p) = 0;
virtual void calcGi(float *p, float *xi) = 0;
virtual void restartDGi() = 0;
virtual void addToDphi(float *device_dphi) = 0;
virtual void configure(int penalizatorIndex, int imageIndex, int imageToAdd) = 0;
void setPenalizationFactor(float p){
        this->penalization_factor = p;
};
void setInu(cufftComplex *Inu){
        this->Inu = Inu;
}
cufftComplex *getInu(){
        return this->Inu;
}
void setS(float *S){
        cudaFree(device_S); this->device_S = S;
};
void setDS(float *DS){
        cudaFree(device_DS); this->device_DS = DS;
};
virtual float calculateSecondDerivate() = 0;
float getPenalizationFactor(){
        return this->penalization_factor;
}
protected:
float *device_S;
float *device_DS;
float penalization_factor = 1;
int imageIndex;
int mod;
int order;
cufftComplex * Inu = NULL;
int imageToAdd;
};


class Image
{
public:
Image(float *image, int image_count){
        this->image = image; this->image_count = image_count;
};
int getImageCount(){
        return image_count;
};
float *getImage(){
        return image;
};
float *getErrorImage(){
        return error_image;
};
imageMap *getFunctionMapping(){
        return functionMapping;
};
void setImageCount(int i){
        this->image_count = i;
};
void setErrorImage(float *f){
        this->error_image = f;
};
void setImage(float *i){
        this->image = i;
};
void setFunctionMapping(imageMap *f){
        this->functionMapping = f;
};
private:
int image_count;
float *image;
float *error_image;
imageMap *functionMapping;
};

class Visibilities
{
public:

void setMSDataset(std::vector<MSDataset> d){
        this->datasets = d;
};
void setTotalVisibilities(int t){
        this->total_visibilities = t;
};

void setNDatasets(int t){
        this->ndatasets = t;
};

void setMaxNumberVis(int t){
        this->max_number_vis = t;
};

std::vector<MSDataset> getMSDataset(){
        return datasets;
};
int getTotalVisibilities(){
        return total_visibilities;
};

int getMaxNumberVis(){
        return max_number_vis;
};

int getNDatasets(){
        return ndatasets;
};

private:
std::vector<MSDataset> datasets;
int ndatasets;
int total_visibilities;
int max_number_vis;
};

class Error
{
public:
virtual void calculateErrorImage(Image *I, Visibilities *v) = 0;
};

class Io
{
public:
virtual canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag) = 0;
virtual void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding) = 0;
virtual void IocopyMS(char const *infile, char const *outfile) = 0;
virtual void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag) = 0;
virtual void IocloseCanvas(fitsfile *canvas) = 0;
virtual void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)= 0;
virtual void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N) = 0;
virtual void IoPrintOptImageIteration(float *I, char *name_image, char *units, int index) = 0;
virtual void IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option)=0;
void setPrintImagesPath(char * pip){
        this->printImagesPath = pip;
};
protected:
int *iteration;
char *printImagesPath;
};

class ObjectiveFunction
{
public:
ObjectiveFunction(){
};
void addFi(Fi *fi){
        if(fi->getPenalizationFactor())
                fis.push_back(fi);
};
//virtual void print() = 0;
float calcFunction(float *p)
{
        float value = 0.0;

        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                value += (*it)->calcFi(p);
        }

        return value;
};

void calcGradient(float *p, float *xi)
{
        if(print_images) {
                if(IoOrderIterations == NULL) {
                        io->IoPrintOptImageIteration(p,"I_nu_0","JY/PIXEL",0);
                        io->IoPrintOptImageIteration(p,"alpha","",1);
                }else{
                        (IoOrderIterations)(p, io);
                }
        }
        restartDPhi();
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->calcGi(p, xi);
                (*it)->addToDphi(dphi);
        }
        phiStatus = 1;
        copyDphiToXi(xi);
};

void restartDPhi()
{
        for(std::vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
        {
                (*it)->restartDGi();
        }
        gpuErrchk(cudaMemset(dphi, 0, sizeof(float)*M*N*image_count));
}

void copyDphiToXi(float *xi)
{
        gpuErrchk(cudaMemcpy(xi, dphi, sizeof(float)*M*N*image_count, cudaMemcpyDeviceToDevice));
}
void setN(long N){
        this->N = N;
}
void setM(long M){
        this->M = M;
}
void setImageCount(int I){
        this->image_count = I;
}
void setIo(Io *i){
        this->io = i;
};
void setPrintImages(int i){
        this->print_images = i;
};
void setIoOrderIterations(void (*func)(float *I, Io *io)){
        this->IoOrderIterations = func;
};
void configure(long N, long M, int I)
{
        setN(N);
        setM(M);
        setImageCount(I);
        gpuErrchk(cudaMalloc((void**)&dphi, sizeof(float)*M*N*I));
        gpuErrchk(cudaMemset(dphi, 0, sizeof(float)*M*N*I));
}
private:
std::vector<Fi*> fis;
Io *io = NULL;
float *dphi;
int phiStatus = 1;
int flag = 0;
long N = 0;
long M = 0;
int print_images = 0;
void (*IoOrderIterations)(float *I, Io *io) = NULL;
int image_count = 1;
};

class Optimizator
{
public:
__host__ virtual void allocateMemoryGpu() = 0;
__host__ virtual void deallocateMemoryGpu() = 0;
__host__ virtual void optimize() = 0;
//__host__ virtual void configure() = 0;
__host__ void setImage(Image *image){
        this->image = image;
};
__host__ void setObjectiveFunction(ObjectiveFunction *of){
        this->of = of;
};
void setFlag(int flag){
        this->flag = flag;
};
ObjectiveFunction* getObjectiveFuntion(){
        return this->of;
};
protected:
ObjectiveFunction *of;
Image *image;
int flag;
};



class Filter
{
public:
virtual void applyCriteria(Visibilities *v) = 0;
virtual void configure(void *params) = 0;
};

//Implementation of Factory
class Synthesizer
{
public:
__host__ virtual void run() = 0;
__host__ virtual void setOutPut(char * FileName) = 0;
__host__ virtual void setDevice() = 0;
__host__ virtual void unSetDevice() = 0;
__host__ virtual std::vector<std::string> countAndSeparateStrings(char *input) = 0;
__host__ virtual void configure(int argc, char **argv) = 0;
__host__ virtual void applyFilter(Filter *filter) = 0;
__host__ void setOptimizator(Optimizator *min){
        this->optimizator = min;
};
__host__ void setVisibilities(Visibilities * v){
        this->visibilities = v;
};
__host__ void setIoHandler(Io *handler){
        this->iohandler = handler;
};
__host__ void setError(Error *e){
        this->error = e;
};
__host__ void setOrder(void (*func)(Optimizator *o,Image *I)){
        this->Order = func;
};
Image *getImage(){
        return image;
};
void setImage(Image *i){
        this->image = i;
};
void setIoOrderEnd(void (*func)(float *I, Io *io)){
        this->IoOrderEnd = func;
};
void setIoOrderError(void (*func)(float *I, Io *io)){
        this->IoOrderError = func;
};
void setIoOrderIterations(void (*func)(float *I, Io *io)){
        this->IoOrderIterations = func;
};
protected:
cufftComplex *device_I;
Image *image;
Optimizator *optimizator;
Io *iohandler = NULL;
Visibilities *visibilities;
Error *error = NULL;
void (*Order)(Optimizator *o, Image *I) = NULL;
int imagesChanged = 0;
void (*IoOrderIterations)(float *I, Io *io) = NULL;
void (*IoOrderEnd)(float *I, Io *io) = NULL;
void (*IoOrderError)(float *I, Io *io) = NULL;
};



class SynthesizerFactory
{
public:
typedef Synthesizer* (*CreateSynthesizerCallback)();
private:
typedef std::map<int, CreateSynthesizerCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterSynthesizer(int SynthesizerId, CreateSynthesizerCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(SynthesizerId, CreateFn)).second;
};

bool UnregisterSynthesizer(int SynthesizerId)
{
        return callbacks_.erase(SynthesizerId) == 1;
};

Synthesizer* CreateSynthesizer(int SynthesizerId)
{
        CallbackMap::const_iterator i = callbacks_.find(SynthesizerId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Synthesizer ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class FiFactory
{
public:
typedef Fi* (*CreateFiCallback)();
private:
typedef std::map<int, CreateFiCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterFi(int FiId, CreateFiCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(FiId, CreateFn)).second;
};

bool UnregisterFi(int FiId)
{
        return callbacks_.erase(FiId) == 1;
};

Fi* CreateFi(int FiId)
{
        CallbackMap::const_iterator i = callbacks_.find(FiId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Fi ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class OptimizatorFactory
{
public:
typedef Optimizator* (*CreateOptimizatorCallback)();
private:
typedef std::map<int, CreateOptimizatorCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterOptimizator(int OptimizatorId, CreateOptimizatorCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(OptimizatorId, CreateFn)).second;
};

bool UnregisterOptimizator(int OptimizatorId)
{
        return callbacks_.erase(OptimizatorId) == 1;
};

Optimizator* CreateOptimizator(int OptimizatorId)
{
        CallbackMap::const_iterator i = callbacks_.find(OptimizatorId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Optimizator ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class FilterFactory
{
public:
typedef Filter* (*CreateFilterCallback)();
private:
typedef std::map<int, CreateFilterCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterFilter(int FilterId, CreateFilterCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(FilterId, CreateFn)).second;
};

bool UnregisterFilter(int FilterId)
{
        return callbacks_.erase(FilterId) == 1;
};

Filter* CreateFilter(int FilterId)
{
        CallbackMap::const_iterator i = callbacks_.find(FilterId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Filter ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class ObjectiveFunctionFactory
{
public:
typedef ObjectiveFunction* (*CreateObjectiveFunctionCallback)();
private:
typedef std::map<int, CreateObjectiveFunctionCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterObjectiveFunction(int ObjectiveFunctionId, CreateObjectiveFunctionCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(ObjectiveFunctionId, CreateFn)).second;
};

bool UnregisterObjectiveFunction(int ObjectiveFunctionId)
{
        return callbacks_.erase(ObjectiveFunctionId) == 1;
};

ObjectiveFunction* CreateObjectiveFunction(int ObjectiveFunctionId)
{
        CallbackMap::const_iterator i = callbacks_.find(ObjectiveFunctionId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown ObjectiveFunction ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class IoFactory
{
public:
typedef Io* (*CreateIoCallback)();
private:
typedef std::map<int, CreateIoCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterIo(int IoId, CreateIoCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(IoId, CreateFn)).second;
};

bool UnregisterIo(int IoId)
{
        return callbacks_.erase(IoId) == 1;
};

Io* CreateIo(int IoId)
{
        CallbackMap::const_iterator i = callbacks_.find(IoId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Io ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

class ErrorFactory
{
public:
typedef Error* (*CreateErrorCallback)();
private:
typedef std::map<int, CreateErrorCallback> CallbackMap;
public:
// Returns true if registration was succesfull
bool RegisterError(int ErrorId, CreateErrorCallback CreateFn)
{
        return callbacks_.insert(CallbackMap::value_type(ErrorId, CreateFn)).second;
};

bool UnregisterError(int ErrorId)
{
        return callbacks_.erase(ErrorId) == 1;
};

Error* CreateError(int ErrorId)
{
        CallbackMap::const_iterator i = callbacks_.find(ErrorId);
        if (i == callbacks_.end())
        {
                // not found
                throw std::runtime_error("Unknown Error ID");
        }
        // Invoke the creation function
        return (i->second)();
};

private:
CallbackMap callbacks_;
};

namespace {
ObjectiveFunction* CreateObjectiveFunction()
{
        return new ObjectiveFunction;
}
const int ObjectiveFunctionId = 0;
const bool Registered = Singleton<ObjectiveFunctionFactory>::Instance().RegisterObjectiveFunction(ObjectiveFunctionId, CreateObjectiveFunction);
};

#endif
