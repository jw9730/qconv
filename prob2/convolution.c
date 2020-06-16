#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

//#define DO_NRMSE

#define Q_CONST 1e2
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

void * quantize(float * S, enum qenum q, int qsize, int num_elem){
    // allocate quantized array
    assert(ALIGN_BYTES % qsize == 0);
    void * Q;
    int rc;
    if ((rc = posix_memalign((void **)&Q, ALIGN_BYTES, num_elem * qsize)) != 0){
        printf("main: quantized memory allocation failure\n");
        exit(-1);
    }
    // amplify, typecast and copy
    for (int i=0; i<num_elem; i++){
        float val = S[i] * Q_CONST;
        if (q == INT32){
            ((int32_t *) Q)[i] = ((int64_t) val > INT32_MAX) ? INT32_MAX : ((int32_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int32_t *) Q)[i], (float) (((int32_t *) Q)[i] / Q_CONST));
        }
        else if (q == INT16){
            ((int16_t *) Q)[i] = ((int64_t) val > INT16_MAX) ? INT16_MAX : ((int16_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int16_t *) Q)[i], (float) (((int16_t *) Q)[i] / Q_CONST));
        }
        else if (q == INT8){
            ((int8_t *) Q)[i] = ((int64_t) val > INT8_MAX) ? INT8_MAX : ((int8_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int8_t *) Q)[i], (float) (((int8_t *) Q)[i] / Q_CONST));
        }
        else continue;
    }
    return Q;
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
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += I[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}

int64_t convolve_quantized(void * I_Q, void * K_Q, int n, int h, int w, int oc, enum qenum q){
    // conditional
    int32_t ret32 = 0;
    int16_t ret16 = 0;
    int8_t ret8 = 0;
    // convolve
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int flag;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                if (q == INT32) {
                    ret32 += ((int32_t *) I_Q)[input_idx] * ((int32_t *) K_Q)[kernel_idx];
                } else if (q == INT16){
                    ret16 += ((int16_t *) I_Q)[input_idx] * ((int16_t *) K_Q)[kernel_idx];
                } else if (q == INT8){
                    ret8 += ((int8_t *) I_Q)[input_idx] * ((int8_t *) K_Q)[kernel_idx];
                } else continue;
            }
        }
    }
    if (q == INT32){
        assert (ret16 == 0 && ret8 == 0);
        return (int64_t) ret32;
    } else if (q == INT16){
        assert (ret32 == 0 && ret8 == 0);
        return (int64_t) ret16;
    } else if (q == INT8){
        assert (ret32 == 0 && ret16 == 0);
        return (int64_t) ret8;
    } else {
        exit(-1);
    }
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
    int qsize = 0;
    if (qbits == 32) {
        q = INT32;
        qsize = sizeof(int32_t);
    }
    else if (qbits == 16) {
        q = INT16;
        qsize = sizeof(int16_t);
    }
    else if (qbits == 8) {
        q = INT8;
        qsize = sizeof(int8_t);
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



    // quantization
    #ifdef DEBUG
    printf("main: quantization bit %d\n", qbits);
    #endif
    clock_t start, end;
    start = clock();
    void * I_Q = quantize(I, q, qsize, N * H * W * C);
    void * K_Q = quantize(K, q, qsize, KH * KW * OC * IC);
    end = clock();
    #ifndef DO_NRMSE
    printf("main: (INT%2.0d, S=%d) -> quantization %f\n", qbits, (int)Q_CONST, ((float) (end - start)) / CLOCKS_PER_SEC);
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
                    if (q == INT32){
                        O[output_idx] = ((float) (int32_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q)) / (Q_CONST * Q_CONST);
                        //printf("main: O[%d,%d,%d,%d]: %d (quantized), %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, ((int32_t *) O_Q)[output_idx], (float)(((int32_t *) O_Q)[output_idx]) / (Q_CONST * Q_CONST), convolve(I, K, n, h, w, oc));
                    }
                    else if (q == INT16){
                        O[output_idx] = ((float) (int16_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q)) / (Q_CONST * Q_CONST);
                        //printf("main: O[%d,%d,%d,%d]: %d (quantized), %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, ((int16_t *) O_Q)[output_idx], (float)(((int16_t *) O_Q)[output_idx]) / (Q_CONST * Q_CONST), convolve(I, K, n, h, w, oc));
                    }
                    else if (q == INT8){
                        O[output_idx] = ((float) (int8_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q)) / (Q_CONST * Q_CONST);
                        //printf("main: O[%d,%d,%d,%d]: %d (quantized), %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, ((int8_t *) O_Q)[output_idx], (float)(((int8_t *) O_Q)[output_idx]) / (Q_CONST * Q_CONST), convolve(I, K, n, h, w, oc));
                    }
                    else continue;
                }
            }
        }
    }
    end = clock();
    #ifndef DO_NRMSE
    printf("main: (INT%2.0d, S=%d) -> convolution %f\n", qbits, (int)Q_CONST, ((float) (end - start)) / CLOCKS_PER_SEC);
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
                    float x;
                    if (q == INT32){
                        x = ((float) ((int32_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q))) / (Q_CONST * Q_CONST);
                    }
                    else if (q == INT16){
                        x = ((float) ((int16_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q))) / (Q_CONST * Q_CONST);
                    }
                    else if (q == INT8){
                        x = ((float) ((int8_t) convolve_quantized(I_Q, K_Q, n, h, w, oc, q))) / (Q_CONST * Q_CONST);
                    }
                    else continue;
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
    free(I_Q); free(K_Q);
    free(I); free(K); free(O);
    return 0;
}