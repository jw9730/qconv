#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))
#define ALIGN_BYTES (sizeof(void *) * 2)
typedef enum qenum{
    INT32,
    INT16,
    INT8
} DATATYPE;

//#define DEBUG
//#define STAT // show data statistics?
#define DO_NRMSE // compute NRMSE?

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
            // clamp overflowing values
            if (((int64_t) val) > INT32_MAX) ((int32_t *) Q)[i] = INT32_MAX;
            else if (((int64_t) val) < INT32_MIN) ((int32_t *) Q)[i] = INT32_MIN;
            else ((int32_t *) Q)[i] = ((int32_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int32_t *) Q)[i], (float) (((int32_t *) Q)[i] / scale));
        }
    }
    else if (q == INT16){
        for (int i=0; i<num_elem; i++){
            float val = S[i] * scale;
            // clamp overflowing values
            if (((int64_t) val) > INT16_MAX) ((int16_t *) Q)[i] = INT16_MAX;
            else if (((int64_t) val) < INT16_MIN) ((int16_t *) Q)[i] = INT16_MIN;
            else ((int16_t *) Q)[i] = ((int16_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int16_t *) Q)[i], (float) (((int16_t *) Q)[i] / scale));
        }
    }
    else if (q == INT8){
        for (int i=0; i<num_elem; i++){
            float val = S[i] * scale;
            // clamp overflowing values
            if (((int64_t) val) > INT8_MAX) ((int8_t *) Q)[i] = INT8_MAX;
            else if (((int64_t) val) < INT8_MIN) ((int8_t *) Q)[i] = INT8_MIN;
            else ((int8_t *) Q)[i] = ((int8_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int8_t *) Q)[i], (float) (((int8_t *) Q)[i] / scale));
        }
    }
    return Q;
}
void quantize_restore(float * O, int64_t * O_Q, int size, float scale2){
    for (int i=0; i<size; i++){
        O[i] = ((float) O_Q[i]) / scale2;
    }
}

void zero_pad(float * PI, float * I, int N, int H, int W, int C, int KH, int KW){
    for (int n=0; n<N; n++){
        for (int ic=0; ic<C; ic++){
            for (int h=0; h<H+KH; h++){
                for (int w=0; w<W+KW; w++){
                    // h, w: position in padded input
                    // position in original input: subtract lower pad
                    int pi_index = INDEX_ROW_MAJOR_4(n,h,w,ic, N,H,W,C);
                    int h_in = h - PH_L;
                    int w_in = w - PW_L;
                    if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) PI[pi_index] = 0;
                    else {
                        int in_index = INDEX_ROW_MAJOR_4(n,h_in,w_in,ic, N,H,W,C);
                        PI[pi_index] = I[in_index];
                    }
                }
            }
        }
    }
}

float convolve(float * PI, float * K, int n, int h, int w, int oc){
    // gets padded input and kernel array, outputs a convolved output value
    // position in padded input
    int h_pad = h + PH_L;
    int w_pad = w + PW_L;
    float ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                input_idx = INDEX_ROW_MAJOR_4(n, h_pad+kh, w_pad+kw, ic, N, H, W, C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += PI[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}
int64_t convolve_q32(void * PI_Q, void * K_Q, int n, int h, int w, int oc){
    int32_t * PI = (int32_t *) PI_Q;
    int32_t * K = (int32_t *) K_Q;
    int h_pad = h + PH_L;
    int w_pad = w + PW_L;
    int64_t ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                input_idx = INDEX_ROW_MAJOR_4(n, h_pad+kh, w_pad+kw, ic, N, H, W, C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += PI[input_idx] * K[kernel_idx]; // implicit typecasting
            }
        }
    }
    return (int64_t) ret;
}
int64_t convolve_q16(void * PI_Q, void * K_Q, int n, int h, int w, int oc){
    int16_t * PI = (int16_t *) PI_Q;
    int16_t * K = (int16_t *) K_Q;
    int h_pad = h + PH_L;
    int w_pad = w + PW_L;
    int32_t ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                input_idx = INDEX_ROW_MAJOR_4(n, h_pad+kh, w_pad+kw, ic, N, H, W, C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += PI[input_idx] * K[kernel_idx]; // implicit typecasting
            }
        }
    }
    return (int64_t) ret;
}
int64_t convolve_q8(void * PI_Q, void * K_Q, int n, int h, int w, int oc){
    int8_t * PI = (int8_t *) PI_Q;
    int8_t * K = (int8_t *) K_Q;
    int h_pad = h + PH_L;
    int w_pad = w + PW_L;
    int16_t ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                input_idx = INDEX_ROW_MAJOR_4(n, h_pad+kh, w_pad+kw, ic, N, H, W, C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += PI[input_idx] * K[kernel_idx]; // implicit typecasting
            }
        }
    }
    return (int64_t) ret;
}


int main(int argc, char **argv){
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: argc=%d\n", argc);
    #endif
    assert (argc == 4);
    // open file
    if (argv[1] == NULL || argv[2] == NULL || argv[3] == NULL) {
        printf("Usage: ./convolution [input_tensor.bin] [kernel_tensor.bin] [32/16/8]\n");
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
    int qbits = atoi(argv[3]);
    enum qenum q;
    int qsize = 0;
    float iscale, kscale;
    int64_t (* qconv) (void *, void *, int, int, int, int) = NULL;
    if (qbits == 32) {
        q = INT32;
        qsize = sizeof(int32_t);
        iscale = (1<<9);
        kscale = (1<<9) * 5e2;
        qconv = &convolve_q32;
    } else if (qbits == 16) {
        q = INT16;
        qsize = sizeof(int16_t);
        iscale = (1<<7);
        kscale = (1<<7) * 5e2;
        qconv = &convolve_q16;
    } else if (qbits == 8) {
        q = INT8;
        qsize = sizeof(int8_t);
        iscale = (1<<0);
        kscale = (1<<0) * 5e2;
        qconv = &convolve_q8;
    } else {
        printf("main: quantization bit should be 32, 16 or 8, got %d\n", qbits);
        exit(-1);
    }
    float scale2 = iscale * kscale;
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////



    ///////////////////////////////////////////read data///////////////////////////////////////////
    // reading metadata
    int isize[4];
    int ksize[4];
    size_t rsize;
    if ((rsize = fread(isize, sizeof(int), 4, ifptr)) != 4){
        printf("main: read failure\n");
        exit(-1);
    }
    if ((rsize = fread(ksize, sizeof(int), 4, kfptr)) != 4){
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
    if ((rsize = fread(I, sizeof(float), N * H * W * C, ifptr)) != N * H * W * C){
        printf("main: read failure\n");
        exit(-1);
    }
    if ((rsize = fread(K, sizeof(float), KH * KW * OC * IC, kfptr)) != KH * KW * OC * IC){
        printf("main: read failure\n");
        exit(-1);
    }
    fclose(ifptr);
    fclose(kfptr);
    #ifdef STAT
    float I_mean = 0;
    float I_std = 0;
    float K_mean = 0;
    float K_std = 0;
    for (int i=0; i<N*H*W*C; i++) I_mean += I[i];
    I_mean /= (N*H*W*C);
    for (int i=0; i<N*H*W*C; i++) I_std += pow(I[i] - I_mean, 2);
    I_std = sqrt(I_std) / (N*H*W*C);
    for (int i=0; i<KH*KW*OC*IC; i++) K_mean += K[i];
    K_mean /= (KH*KW*OC*IC);
    for (int i=0; i<KH*KW*OC*IC; i++) K_std += pow(K[i] - K_mean, 2);
    K_std = sqrt(K_std) / (KH*KW*OC*IC);
    printf("STATISTICS:\t\tinput %.8f+-%.8f, kernel %.8f+-%.8f\n", I_mean, I_std, K_mean, K_std);
    printf("SCALED STATISTICS:\tinput %.8f+-%.8f, kernel %.8f+-%.8f\n", I_mean*iscale, I_std*iscale, K_mean*kscale, K_std*kscale);
    #endif
    ///////////////////////////////////////////read data///////////////////////////////////////////



    ///////////////////////////////////////////zero pad///////////////////////////////////////////
    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;
    // declared padded input array
    float * PI;
    if ((rc = posix_memalign((void **)&PI, ALIGN_BYTES, N * (H+KH) * (W+KW) * C * sizeof(float))) != 0){
        printf("main: input memory allocation failure\n");
        exit(-1);
    }
    zero_pad(PI, I, N, H, W, C, KH, KW);
    ///////////////////////////////////////////zero pad///////////////////////////////////////////



    ///////////////////////////////////////////quantization///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: quantization bit %d\n", qbits);
    #endif
    clock_t start, end;
    start = clock();
    void * PI_Q = quantize(PI, q, qsize, iscale, N * (H+KH) * (W+KW) * C);
    void * K_Q = quantize(K, q, qsize, kscale, KH * KW * OC * IC);
    int64_t * O_Q;
    if ((rc = posix_memalign((void **)&O_Q, ALIGN_BYTES, N * H * W * OC * sizeof(int64_t))) != 0){
        printf("main: output memory allocation failure\n");
        exit(-1);
    }
    end = clock();
    float qtime = ((float) (end - start)) / CLOCKS_PER_SEC;
    ///////////////////////////////////////////quantization///////////////////////////////////////////



    ///////////////////////////////////////////main routine///////////////////////////////////////////
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
                    int output_idx = INDEX_ROW_MAJOR_4(n,h,w,oc, N,H,W,OC);
                    O_Q[output_idx] = qconv(PI_Q, K_Q, n, h, w, oc);
                    //if (oc==0 && h==H/2) printf("main: O[%d,%d,%d,%d]: %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, ((float)O_Q[output_idx])/scale2, convolve(PI, K, n, h, w, oc));
                }
            }
        }
    }
    end = clock();
    float ctime = ((float) (end - start)) / CLOCKS_PER_SEC;
    ///////////////////////////////////////////main routine///////////////////////////////////////////



    ///////////////////////////////////////////postprocessing///////////////////////////////////////////
    start = clock();
    quantize_restore(O, O_Q, N*H*W*OC, scale2);
    end = clock();
    qtime += ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("INT%2.0d convolution %fs, overhead %fs\n", qbits, ctime, qtime);
    ///////////////////////////////////////////postprocessing///////////////////////////////////////////



    ///////////////////////////////////////////precision analysis///////////////////////////////////////////
    #ifdef DO_NRMSE
    float ymax = -1e15;
    float ymin = 1e15;
    float acc = 0;
    for (int n=0; n<N; n++){
        for (int h=0; h<H; h++){
            for (int w=0; w<W; w++){
                for (int oc=0; oc<OC; oc++){
                    float x = O[INDEX_ROW_MAJOR_4(n,h,w,oc, N,H,W,OC)];
                    float y = convolve(PI, K, n, h, w, oc);
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
    printf("main: flush output to file, header [%d,%d,%d,%d]: ", N, H, W, OC);
    if ((ofptr = fopen("output_tensor.bin", "wb")) == NULL){
        printf("output file open failed\n");
        exit(-1);
    }
    int32_t header[4] = {N, H, W, OC};
    if ((rsize = fwrite(header, sizeof(int32_t), 4, ofptr)) != 4){
        printf("main: write failure\n");
        exit(-1);
    }
    if ((rsize = fwrite(O, sizeof(float), N * H * W * OC, ofptr)) !=  N * H * W * OC){
        printf("main: write failure\n");
        exit(-1);
    }
    fclose(ofptr);
    // check correctness
    if ((ofptr = fopen("output_tensor.bin", "rb")) == NULL){
        printf("output file open failed\n");
        exit(-1);
    }
    if ((rsize = fread(header, sizeof(int), 4, ofptr)) != 4){
        printf("main: read failure\n");
        exit(-1);
    }
    fclose(ofptr);
    printf("retrieved [%d,%d,%d,%d]\n", header[0], header[1], header[2], header[3]);
    free(PI_Q); free(K_Q);
    free(I); free(K); free(O); free(PI);
    return 0;
    ///////////////////////////////////////////tidying up///////////////////////////////////////////
}