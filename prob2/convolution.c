#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#define INDEX_ROW_MAJOR_2(i, j, I, J) ((j) + (J) * (i))
#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) ((k) + (K) * ((j) + (J) * (i)))
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))
#define ALIGN_BYTES (sizeof(void *) * 2)

//#define DEBUG
//#define CHECK_OVERFLOW
//int max_size = ((qtype)~(qtype)0);

#define DO_NRMSE

#define Q_CONST 1e4
#define Q_CONST2 1e8
int qbits;
typedef enum qenum{
    INT32,
    INT16,
    INT8
} DATATYPE; 

FILE * ifptr, * kfptr, * ofptr;
int N, H, W, C;
int KH, KW, OC, IC;
int PH_L, PH_H, PW_L, PW_H;

void * quantize(float * S, enum qenum q, int num_elem){
    // parse quantization type
    typedef int64_t qtype;
    if (q == INT32) {typedef int32_t qtype;}
    else if (q == INT16) {typedef int16_t qtype;}
    else if (q == INT8) {typedef int8_t qtype;}
    else (assert(0));
    qtype * Q;
    // allocate quantized array
    assert(ALIGN_BYTES % sizeof(qtype) == 0);
    int rc;
    if ((rc = posix_memalign((void **)&Q, ALIGN_BYTES, num_elem * sizeof(qtype))) != 0){
        printf("main: quantized memory allocation failure\n");
        exit(-1);
    }
    // amplify, typecast and copy
    for (int i=0; i<num_elem; i++){
        Q[i] = (qtype) (S[i] * Q_CONST);
        //printf("quantize: %f -> %d -> %f\n", S[i], (int)Q[i], (float) (Q[i] / Q_CONST));
    }
    return Q;
}

void restore_product(float * S, float * Q, enum qenum q, int num_elem){
    // restore quantization
    for (int i=0; i<num_elem; i++){
        S[i] = (float) (Q[i] / Q_CONST2);
        //printf("restore: %d -> %f\n", Q[i], S[i]);
    }
}

float convolve(float * I, float * K, int n, int h, int w, int oc){
    // gets input and kernel array of same size, outputs a convolved output value, assume zero padding
    // (padded) input boundary corresponding to window
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    float ret = 0;
    int flag;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, OC);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += I[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}

int64_t convolve_quantized(void * I_Q, void * K_Q, int n, int h, int w, int oc, enum qenum q){
    // parse quantization type
    typedef int64_t qtype;
    if (q == INT32) {typedef int32_t qtype;}
    else if (q == INT16) {typedef int16_t qtype;}
    else if (q == INT8) {typedef int8_t qtype;}
    qtype * I_qtype = (qtype *) I_Q;
    qtype * K_qtype = (qtype *) K_Q;
    // convolve
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    qtype ret = 0;
    int flag;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, OC);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += I_qtype[input_idx] * K_qtype[kernel_idx];
            }
        }
    }
    return (int64_t) ret;
}

int main(int argc, char **argv){
    #ifdef DEBUG
    printf("main: argc=%d\n", argc);
    #endif
    assert (argc == 4);
    // open file
    if (argv[1] == NULL || argv[2] == NULL || argv[3] == NULL) {
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
    // quantization bits
    qbits = atoi(argv[3]);
    enum qenum q;
    typedef float qtype;
    if (qbits == 32) {
        q = INT32;
        typedef int32_t qtype;
    }
    else if (qbits == 16) {
        q = INT16;
        typedef int16_t qtype;
    }
    else if (qbits == 8) {
        q = INT8;
        typedef int8_t qtype;
    }
    else {
        printf("main: quantization bit should be 32, 16 or 8, got %d\n", qbits);
        exit(-1);
    }

    // reading metadata
    int isize[4];
    int ksize[4];
    fread(isize, sizeof(int), 4, ifptr);
    fread(ksize, sizeof(int), 4, kfptr);
    #ifdef DEBUG
    printf("main: (N, H, W, C) = (%d, %d, %d, %d)\n", isize[0], isize[1], isize[2], isize[3]);
    printf("main: (KH, KW, OC, IC) = (%d, %d, %d, %d)\n", ksize[0], ksize[1], ksize[2], ksize[3]);
    #endif

    N = isize[0]; H = isize[1]; W = isize[2]; C = isize[3];
    KH = ksize[0]; KW = ksize[1]; OC = ksize[2]; IC = ksize[3];
    assert(C == IC);

    #ifdef DEBUG
    printf("main: read input and kernel file into memory\n");
    #endif
    // allocate
    float * I, * K, * O;
    assert(ALIGN_BYTES % sizeof(float) == 0);
    int rc;
    if ((rc = posix_memalign((void **)&I, ALIGN_BYTES, N * H * W * C * sizeof(float))) != 0){
        printf("main: input memory allocation failure\n");
        exit(-1);
    }
    if ((rc = posix_memalign((void **)&K, ALIGN_BYTES, KH * KW * OC * IC * sizeof(float))) != 0){
        printf("main: kernel memory allocation failure\n");
        exit(-1);
    }
    if ((rc = posix_memalign((void **)&O, ALIGN_BYTES, N * H * W * OC * sizeof(float))) != 0){
        printf("main: output memory allocation failure\n");
        exit(-1);
    }
    #ifdef DEBUG
    printf("main: I %p, K %p, O %p, align_bytes %lu, sizeof(float) %lu\n", I, K, O, ALIGN_BYTES, sizeof(float));
    #endif
    // read file into memory
    fread(I, sizeof(float), N * H * W * C, ifptr);
    fread(K, sizeof(float), KH * KW * OC * IC, kfptr);
    fclose(ifptr);
    fclose(kfptr);

    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;



    // quantization routine
    clock_t start, end;
    start = clock();
    qtype * O_Q;
    assert(ALIGN_BYTES % sizeof(qtype) == 0);
    if ((rc = posix_memalign((void **)&O_Q, ALIGN_BYTES, N * H * W * OC * sizeof(qtype))) != 0){
        printf("main: output memory allocation failure\n");
        exit(-1);
    }
    qtype * I_Q = quantize(I, q, N * H * W * C);
    qtype * K_Q = quantize(K, q, KH * KW * OC * IC);
    end = clock();
    #ifndef DO_NRMSE
    printf("main: quantization elapsed time: %f\n", ((float) (end - start)) / CLOCKS_PER_SEC);
    #endif


    #ifdef DEBUG
    printf("main: compute convolution into output @ %p\n", O);
    #endif
    start = clock();
    
    // compute convolution (scalar operations)
    for (int n=0; n<N; n++){
        for (int h=0; h<H; h++){
            for (int w=0; w<W; w++){
                for (int oc=0; oc<OC; oc++){
                    // convolution for a single output pixel
                    int output_idx = INDEX_ROW_MAJOR_4(n, h, w, oc, N, H, W, OC);
                    O_Q[output_idx] = (qtype) convolve_quantized(I_Q, K_Q, n, h, w, oc, q);
                    //printf("main: O[%d,%d,%d,%d]: %d (quantized), %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, (int) O_Q[output_idx], (float)(O_Q[output_idx] / (qtype) ((int) Q_CONST2)), convolve(I, K, n, h, w, oc));
                }
            }
        }
    }
    restore_product(O, O_Q, q, N * H * W * OC);

    end = clock();
    #ifndef DO_NRMSE
    printf("main: convolution elapsed time: %f\n", ((float) (end - start)) / CLOCKS_PER_SEC);
    #endif


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
                    float x = (float) (((qtype) convolve_quantized(I_Q, K_Q, n, h, w, oc, q)) / Q_CONST2);
                    float y = convolve(I, K, n, h, w, oc);
                    acc += (x - y)*(x - y);
                    if (ymax<y) ymax = y;
                    if (ymin>y) ymin = y;
                }
            }
        }
    }
    float NRMSE = sqrt(acc / (N*H*W*OC))/(ymax-ymin);
    printf("main: (INT%2.0d, S=%d) -> NRMSE=%.20f\n", qbits, (int)Q_CONST, NRMSE);
    #endif
    ///////////////////////////////////////////precision analysis///////////////////////////////////////////



    // make output file
    #ifdef DEBUG
    printf("main: flush output to file\n");
    #endif
    if ((ofptr = fopen("output_tensor.bin", "wb")) == NULL){
        printf("output file open failed\n");
        exit(-1);
    }
    fwrite(O, sizeof(float), N * H * W * OC, ofptr);
    fclose(ofptr);
    free(I_Q); free(K_Q); free(O_Q);
    free(I); free(K); free(O);
    return 0;
}