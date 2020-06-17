#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) ((k) + (K) * ((j) + (J) * (i)))
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))
#define THREADS_PER_BLOCK 512
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

//#define DEBUG
#define DO_NRMSE

FILE * ifptr, * kfptr, * ofptr;
int N, H, W, C;
int KH, KW, OC, IC;
int PH_L, PH_H, PW_L, PW_H;

float convolve(float * I, float * K, int n, int h, int w, int oc){
    // gets input and kernel array of same size, outputs a convolved output value, assume zero padding
    // (padded) input boundary corresponding to window
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    float ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
                input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += I[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}
__global__ void convolve_cuda(float *I, float *K, float *O, int N, int H, int W, int KH, int KW, int IC, int OC, int PH_L, int PW_L){
    // input stationary
    int BLOCKS_PER_PIXEL = ceil((float)(OC)/(float)(THREADS_PER_BLOCK));
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int cid = bid % BLOCKS_PER_PIXEL; // channel block index (within pixel)
    int pid = bid / BLOCKS_PER_PIXEL; // pixel index (N, H, W)
    // parse pixel index
    int n = pid/(H*W);
    int h = pid%(H*W)/W;
    int w = pid%W;
    // (padded) input boundary corresponding to window
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    // declare on-chip shared memory
    extern __shared__ float M[];
    // read input data once per block (shared across threads)
    // this process could serve as bottleneck, load distribution is critical
    // distribute indices across threads
    int shm_size = KH * KW * IC;
    int shm_per_t = ceil((float)(shm_size)/(float)(THREADS_PER_BLOCK));
    int l = shm_per_t * tid;
    int u = shm_per_t * (tid + 1);
    // parse idx (KH, KW, IC)
    if (l < shm_size) {
        for (int idx=l; idx<((u<shm_size)?u:shm_size); idx++){
            int kh = idx/(KW*IC);
            int kw = idx%(KW*IC)/IC;
            int ic = idx%IC;
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) M[INDEX_ROW_MAJOR_3(kh,kw,ic, KH,KW,IC)] = 0;
            else M[INDEX_ROW_MAJOR_3(kh,kw,ic, KH,KW,IC)] = I[INDEX_ROW_MAJOR_4(n,IH_L+kh,IW_L+kw,ic, N,H,W,IC)];
        }
    }
    // wait until data is ready
    __syncthreads();
    // compute block index in output channel dimension
    int ofs = cid * THREADS_PER_BLOCK;
    // handle boundary
    if (tid >= ((OC - ofs < THREADS_PER_BLOCK)? (OC - ofs) : THREADS_PER_BLOCK)) return;
    // apply convolution
    float acc = 0;
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            for (int ic=0; ic<IC; ic++){
                acc += M[INDEX_ROW_MAJOR_4(n,kh,kw,ic, N,KH,KW,IC)] * K[INDEX_ROW_MAJOR_4(kh,kw,ofs+tid,ic, KH,KW,OC,IC)];
            }
        }
    }
    O[INDEX_ROW_MAJOR_4(n,h,w,ofs+tid, N,H,W,OC)] = acc;
}
int main(int argc, char **argv){
    printf("\n");
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: argc=%d\n", argc);
    #endif
    assert (argc == 3);
    // open file
    if (argv[1] == NULL || argv[2] == NULL) {
        printf("Usage: ./convolution [input_tensor.bin] [kernel_tensor.bin]\n");
        exit(-1);
    }
    if ((ifptr = fopen(argv[1], "rb")) == NULL){
        printf("invalid input file\n");
        exit(-1);
    }
    if ((kfptr = fopen(argv[2], "rb")) == NULL){
        printf("invalid kernel file\n");
        exit(-1);
    }
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////



    ///////////////////////////////////////////read data///////////////////////////////////////////
    // reading metadata
    int isize[4];
    int ksize[4];
    size_t rsize;
    if ((rsize = fread(isize, sizeof(int), 4, ifptr)) != 4 * sizeof(int)){
        printf("main: read failure\n");
        exit(-1);
    }
    if ((rsize = fread(ksize, sizeof(int), 4, kfptr)) != 4 * sizeof(int)){
        printf("main: read failure\n");
        exit(-1);
    }
    #ifdef DEBUG
    printf("main: (N, H, W, C) = (%d, %d, %d, %d)\n", isize[0], isize[1], isize[2], isize[3]);
    printf("main: (KH, KW, OC, IC) = (%d, %d, %d, %d)\n", ksize[0], ksize[1], ksize[2], ksize[3]);
    #endif
    N = isize[0]; H = isize[1]; W = isize[2]; C = isize[3];
    KH = ksize[0]; KW = ksize[1]; OC = ksize[2]; IC = ksize[3];
    assert(C == IC);
    // reading data
    #ifdef DEBUG
    printf("main: read input and kernel file into memory\n");
    #endif
    float * I, * K, * O;
    size_t align_bytes = sizeof(void *) * 2;
    assert(align_bytes % sizeof(float) == 0);
    int rc;
    if ((rc = posix_memalign((void **)&I, align_bytes, N * H * W * C * sizeof(float))) != 0){
        printf("main: input memory allocation failure\n");
        exit(-1);
    }
    if ((rc = posix_memalign((void **)&K, align_bytes, KH * KW * OC * IC * sizeof(float))) != 0){
        printf("main: kernel memory allocation failure\n");
        exit(-1);
    }
    if ((rc = posix_memalign((void **)&O, align_bytes, N * H * W * OC * sizeof(float))) != 0){
        printf("main: output memory allocation failure\n");
        exit(-1);
    }
    #ifdef DEBUG
    printf("main: I %p, K %p, O %p, align_bytes %lu, sizeof(float) %lu\n", I, K, O, align_bytes, sizeof(float));
    #endif
    // read file into memory
    // read file into memory
    if ((rsize = fread(I, sizeof(float), N * H * W * C, ifptr)) != N * H * W * C * sizeof(float)){
        printf("main: read failure\n");
        exit(-1);
    }
    if ((rsize = fread(K, sizeof(float), KH * KW * OC * IC, kfptr)) != KH * KW * OC * IC * sizeof(float)){
        printf("main: read failure\n");
        exit(-1);
    }
    fclose(ifptr);
    fclose(kfptr);
    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;
    float *dev_I, *dev_K, *dev_O;
    ///////////////////////////////////////////read data///////////////////////////////////////////



    ///////////////////////////////////////////device setup///////////////////////////////////////////
    // loop over outer dimensions, and compute dot product in chunks of size 512
    // kernel function: convolution for a single sliding window
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_I, N * H * W * C * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_K, H * W * OC * IC * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_O, N * H * W * OC * sizeof(float) ) );
    // copy the arrays to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_I, I, N * H * W * C * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_K, K, KH * KW * OC * IC * sizeof(float), cudaMemcpyHostToDevice ) );
    // how to organize blocks?
    // maximizing data reuse and parallelism within a block
    int BLOCK_MEMSIZE = KH * KW * IC * sizeof(float);
    // input stationary
    // within a block, hold input and thread over output channels
    int BLOCKS_PER_PIXEL = ceil((float)(OC)/(float)(THREADS_PER_BLOCK));
    ///////////////////////////////////////////device setup///////////////////////////////////////////



    ///////////////////////////////////////////main routine///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: compute convolution into output @ %p\n", O);
    #endif
    clock_t start, end;
    start = clock();
    convolve_cuda<<<N * H * W * BLOCKS_PER_PIXEL,THREADS_PER_BLOCK,BLOCK_MEMSIZE>>>(dev_I, dev_K, dev_O, N, H, W, KH, KW, IC, OC, PH_L, PW_L);
    // copy the array back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( O, dev_O, N * H * W * OC * sizeof(float), cudaMemcpyDeviceToHost ) );
    end = clock();
    float ctime = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("convolution %fs\n", ctime);
    ///////////////////////////////////////////main routine///////////////////////////////////////////



    ///////////////////////////////////////////precision analysis///////////////////////////////////////////
    #ifdef DO_NRMSE
    // metric: NRMSE
    float ymax = -1e15;
    float ymin = 1e15;
    float acc = 0;
    for (int n=0; n<N; n++){
        for (int h=0; h<H; h++){
            for (int w=0; w<W; w++){
                for (int oc=0; oc<OC; oc++){
                    int output_idx = INDEX_ROW_MAJOR_4(n, h, w, oc, N, H, W, OC);
                    float x = O[output_idx];
                    float y = convolve(I, K, n, h, w, oc);
                    acc += (x - y)*(x - y) / (N*H*W*OC);
                    if (ymax<y) ymax = y;
                    if (ymin>y) ymin = y;
                }
            }
        }
    }
    float NRMSE = sqrt(acc)/(ymax-ymin);
    printf("NRMSE=%.20f\n", NRMSE);
    #endif
    ///////////////////////////////////////////precision analysis///////////////////////////////////////////



    ///////////////////////////////////////////tidying up///////////////////////////////////////////
    // make output file
    #ifdef DEBUG
    printf("main: flush output to file\n");
    #endif
    if ((ofptr = fopen("output_tensor.bin", "wb")) == NULL){
        printf("output file open failed\n");
        exit(-1);
    }
    if ((rsize = fwrite(O, sizeof(float), N * H * W * OC, ofptr)) !=  N * H * W * OC * sizeof(float)){
        printf("main: write failure\n");
        exit(-1);
    }
    fclose(ofptr);
    free(I); free(K); free(O);
    cudaFree(dev_I); cudaFree(dev_K); cudaFree(dev_O);
    return 0;
    ///////////////////////////////////////////tidying up///////////////////////////////////////////
}