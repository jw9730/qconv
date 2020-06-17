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
#define DO_NRMSE

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

void * quantize(float * S, enum qenum q, int qsize, float scale, int num_elem){
    // allocate quantized array
    assert(ALIGN_BYTES % qsize == 0);
    void * Q;
    int rc;
    if ((rc = posix_memalign((void **)&Q, ALIGN_BYTES, num_elem * qsize)) != 0){
        printf("main: quantized memory allocation failure\n");
        exit(-1);
    }
    // amplify, typecast and copy
    if (q == INT32){
        for (int i=0; i<num_elem; i++){
            float val = S[i] * scale;
            if (((int64_t) val) > INT32_MAX) ((int32_t *) Q)[i] = INT32_MAX;
            else if (((int64_t) val) < INT32_MIN) ((int32_t *) Q)[i] = INT32_MIN;
            else ((int32_t *) Q)[i] = ((int32_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int32_t *) Q)[i], (float) (((int32_t *) Q)[i] / scale));
        }
    }
    else if (q == INT16){
        for (int i=0; i<num_elem; i++){
            float val = S[i] * scale;
            if (((int64_t) val) > INT16_MAX) ((int16_t *) Q)[i] = INT16_MAX;
            else if (((int64_t) val) < INT16_MIN) ((int16_t *) Q)[i] = INT16_MIN;
            else ((int16_t *) Q)[i] = ((int16_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int16_t *) Q)[i], (float) (((int16_t *) Q)[i] / scale));
        }
    }
    else if (q == INT8){
        for (int i=0; i<num_elem; i++){
            float val = S[i] * scale;
            if (((int64_t) val) > INT8_MAX) ((int8_t *) Q)[i] = INT8_MAX;
            else if (((int64_t) val) < INT8_MIN) ((int8_t *) Q)[i] = INT8_MIN;
            else ((int8_t *) Q)[i] = ((int8_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int8_t *) Q)[i], (float) (((int8_t *) Q)[i] / scale));
        }
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
float convolve_quantized8(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    // convolve
    int8_t * I = (int8_t *) I_Q;
    int8_t * K = (int8_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int flag;
    float ret = 0;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                //printf("%f <- %d = %d * %d\n", ((float) (I[input_idx] * K[kernel_idx])) / scale2, I[input_idx] * K[kernel_idx], I[input_idx], K[kernel_idx]);
                ret += ((float) (I[input_idx] * K[kernel_idx]));
            }
        }
    }
    return ret;
}
float convolve_quantized16(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    // convolve
    int16_t * I = (int16_t *) I_Q;
    int16_t * K = (int16_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int flag;
    float ret = 0;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                //printf("%f <- %d = %d * %d\n", ((float) (I[input_idx] * K[kernel_idx])) / scale2, I[input_idx] * K[kernel_idx], I[input_idx], K[kernel_idx]);
                ret += ((float) (I[input_idx] * K[kernel_idx]));
            }
        }
    }
    return ret;
}
float convolve_quantized32(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    // convolve
    int32_t * I = (int32_t *) I_Q;
    int32_t * K = (int32_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int flag;
    float ret = 0;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag) continue;
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                //printf("%f <- %d = %d * %d\n", ((float) (I[input_idx] * K[kernel_idx])) / scale2, I[input_idx] * K[kernel_idx], I[input_idx], K[kernel_idx]);
                ret += ((float) (I[input_idx] * K[kernel_idx]));
            }
        }
    }
    return ret;
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
    float scale;
    float (* qconv) (void *, void *, int, int, int, int) = NULL;
    if (qbits == 32) {
        q = INT32;
        qsize = sizeof(int32_t);
        scale = 1<<13;
        assert (INT32_MAX >= (scale * scale));
        qconv = &convolve_quantized32;

    } else if (qbits == 16) {
        q = INT16;
        qsize = sizeof(int16_t);
        scale = 1<<7;
        assert (INT16_MAX >= (scale * scale));
        qconv = &convolve_quantized16;
    } else if (qbits == 8) {
        q = INT8;
        qsize = sizeof(int8_t);
        scale = 1<<3;
        assert (INT8_MAX >= (scale * scale));
        qconv = &convolve_quantized8;
    } else {
        printf("main: quantization bit should be 32, 16 or 8, got %d\n", qbits);
        exit(-1);
    }
    int scale2 = scale * scale;

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
    void * I_Q = quantize(I, q, qsize, scale, N * H * W * C);
    void * K_Q = quantize(K, q, qsize, scale, KH * KW * OC * IC);
    end = clock();
    float qtime = ((float) (end - start)) / CLOCKS_PER_SEC;


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
                    O[output_idx] = qconv(I_Q, K_Q, n, h, w, oc) / scale2;
                    //if (oc==0) printf("main: O[%d,%d,%d,%d]: %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, O[output_idx], convolve(I, K, n, h, w, oc));
                }
            }
        }
    }
    end = clock();
    float ctime = ((float) (end - start)) / CLOCKS_PER_SEC;
    #ifndef DO_NRMSE
    printf("main: (INT%2.0d, S=%.3f) -> quantize %f, convolution %f\n", qbits, scale, qtime, ctime);
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
                    float x = qconv(I_Q, K_Q, n, h, w, oc) / scale2;
                    float y = convolve(I, K, n, h, w, oc);
                    acc += (x - y)*(x - y) / (N*H*W*OC);
                    if (ymax<y) ymax = y;
                    if (ymin>y) ymin = y;
                }
            }
        }
    }
    float NRMSE = sqrt(acc)/(ymax-ymin);
    printf("main: (INT%2.0d, S=%.3f) -> NRMSE=%.20f\n", qbits, scale, NRMSE);
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