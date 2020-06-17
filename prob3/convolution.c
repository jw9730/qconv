#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))
#define ALIGN_BYTES (sizeof(void *) * 4)
typedef enum qenum{
    FP32,
    INT32,
    INT16
} DATATYPE;

// compile flags
//#define DEBUG
//#define DO_NRMSE
//#define PRECISION

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
            if (((int32_t) val) > INT16_MAX) ((int16_t *) Q)[i] = INT16_MAX;
            else if (((int32_t) val) < INT16_MIN) ((int16_t *) Q)[i] = INT16_MIN;
            else ((int16_t *) Q)[i] = ((int16_t) val);
            //printf("quantize: %d, %f -> %d -> %f\n", q, S[i], ((int16_t *) Q)[i], (float) (((int16_t *) Q)[i] / scale));
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
float convolve_avx_fp32(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    float * I = (float *) I_Q;
    float * K = (float *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    float ret = 0;
    // parallelize multiplication with avx
    // I: row major, (N, H, W, IC)
    // K: row major, (KH, KW, OC, IC)
    // O: row major, (N, H, W, OC)
    // avx vectorization dimension: IC
    __m256 acc = _mm256_setzero_ps();
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
            int ic = 0;
            int residue = IC;
            float * I_p = I + INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
            float * K_p = K + INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
            for (int chunk=0; chunk<IC/8; chunk++){
                __m256 vx = _mm256_load_ps(I_p);
                __m256 vy = _mm256_load_ps(K_p);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vy));
                ic += 8; residue -= 8; I_p += 8; K_p += 8;
            }
            // handle boundary
            __m256 vx = _mm256_setzero_ps();
            __m256 vy = _mm256_setzero_ps();
            memcpy(&vx, I_p, sizeof(float) * residue);
            memcpy(&vy, K_p, sizeof(float) * residue);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vx, vy));
        }
    }
    for (int k=0; k<8; k++) ret += ((float *)&acc)[k];
    return ret;
}
float convolve_avx_int32(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    int32_t * I = (int32_t *) I_Q;
    int32_t * K = (int32_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    float ret = 0;
    // parallelize multiplication with avx
    // I: row major, (N, H, W, IC)
    // K: row major, (KH, KW, OC, IC)
    // O: row major, (N, H, W, OC)
    // avx vectorization dimension: IC
    __m256 acc = _mm256_setzero_ps();
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
            int ic = 0;
            int residue = IC;
            int32_t * I_p = I + INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
            int32_t * K_p = K + INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
            for (int chunk=0; chunk<IC/8; chunk++){
                __m256i vx = _mm256_load_si256((__m256i *)I_p);
                __m256i vy = _mm256_load_si256((__m256i *)K_p);
                acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_mullo_epi32(vx, vy)));
                ic += 8; residue -= 8; I_p += 8; K_p += 8;
            }
            // handle boundary
            __m256i vx = (__m256i) _mm256_setzero_ps();
            __m256i vy = (__m256i) _mm256_setzero_ps();
            memcpy(&vx, I_p, sizeof(int32_t) * residue);
            memcpy(&vy, K_p, sizeof(int32_t) * residue);
            acc = _mm256_add_ps(acc, _mm256_cvtepi32_ps(_mm256_mullo_epi32(vx, vy)));
        }
    }
    for (int k=0; k<8; k++) ret += ((float *)&acc)[k];
    return ret;
}
float convolve_avx_int16(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    int16_t * I = (int16_t *) I_Q;
    int16_t * K = (int16_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    float ret = 0;
    // parallelize multiplication with avx
    // I: row major, (N, H, W, IC)
    // K: row major, (KH, KW, OC, IC)
    // O: row major, (N, H, W, OC)
    // avx vectorization dimension: IC
    __m256 acc = _mm256_setzero_ps();
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
            int ic = 0;
            int residue = IC;
            int16_t * I_p = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
            int16_t * K_p = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
            for (int chunk=0; chunk<IC/16; chunk++){
                __m256i vx = _mm256_load_si256((__m256i *)I_p);
                __m256i vy = _mm256_load_si256((__m256i *)K_p);
                #ifdef PRECISION
                // expand to two 32-bits (for precision)
                __m128i xl = _mm_load_si128((__m128i *)&vx);
                __m128i xh = _mm_load_si128(((__m128i *)&vx)+1);
                __m128i yl = _mm_load_si128((__m128i *)&vy);
                __m128i yh = _mm_load_si128(((__m128i *)&vy)+1);
                // compute product
                __m256i vo_l = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(xl), _mm256_cvtepi16_epi32(yl));
                __m256i vo_h = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(xh), _mm256_cvtepi16_epi32(yh));
                acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_cvtepi32_ps(vo_h), _mm256_cvtepi32_ps(vo_l)));
                #endif
                #ifndef PRECISION
                __m256i vo = _mm256_mullo_epi16(vx, vy);
                // converting to 32-bit fp
                __m256i lo_32 = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)&vo));
                __m256i hi_32 = _mm256_cvtepi16_epi32(_mm_load_si128(((__m128i *)&vo)+1));
                // cast to float and accumulate
                acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_cvtepi32_ps(lo_32), _mm256_cvtepi32_ps(hi_32)));
                #endif
                ic += 16; residue -= 16; I_p += 16; K_p += 16;
            }
            // handle boundary
            __m256i vx = (__m256i) _mm256_setzero_ps();
            __m256i vy = (__m256i) _mm256_setzero_ps();
            memcpy(&vx, I_p, sizeof(int16_t) * residue);
            memcpy(&vy, K_p, sizeof(int16_t) * residue);
            #ifdef PRECISION
            // expand to two 32-bits (for precision)
            __m128i xl = _mm_load_si128((__m128i *)&vx);
            __m128i xh = _mm_load_si128(((__m128i *)&vx)+1);
            __m128i yl = _mm_load_si128((__m128i *)&vy);
            __m128i yh = _mm_load_si128(((__m128i *)&vy)+1);
            // compute product
            __m256i vo_l = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(xl), _mm256_cvtepi16_epi32(yl));
            __m256i vo_h = _mm256_mullo_epi32(_mm256_cvtepi16_epi32(xh), _mm256_cvtepi16_epi32(yh));
            acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_cvtepi32_ps(vo_h), _mm256_cvtepi32_ps(vo_l)));
            #endif
            #ifndef PRECISION
            __m256i vo = _mm256_mullo_epi16(vx, vy);
            // converting to 32-bit fp
            __m256i lo_32 = _mm256_cvtepi16_epi32(_mm_load_si128((__m128i *)&vo));
            __m256i hi_32 = _mm256_cvtepi16_epi32(_mm_load_si128(((__m128i *)&vo)+1));
            // cast to float and accumulate
            acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_cvtepi32_ps(lo_32), _mm256_cvtepi32_ps(hi_32)));
            #endif
        }
    }
    for (int k=0; k<8; k++) ret += ((float *)&acc)[k];
    return ret;
}



int main(int argc, char **argv){
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: argc=%d\n", argc);
    #endif
    assert (argc == 4);
    // open file
    if (argv[1] == NULL || argv[2] == NULL || argv[3] == NULL) {
        printf("Usage: ./convolution [input_tensor.bin] [kernel_tensor.bin] [FP32/INT32/INT16]\n");
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
    // parse quantization option
    char str1[10] = "FP32";
    char str2[10] = "INT32";
    char str3[10] = "INT16";
    float (* qconv) (void *, void *, int, int, int, int) = NULL;
    enum qenum q;
    int qbits = 0;
    int qsize = 0;
    float scale = 0;
    if (strcmp(argv[3], str1) == 0){
        q = FP32;
        qsize = sizeof(float);
        qconv = &convolve_avx_fp32;
        scale = 1;
    } else if (strcmp(argv[3], str2) == 0){
        q = INT32;
        qbits = 32;
        qsize = sizeof(int32_t);
        scale = 1<<13;
        qconv = &convolve_avx_int32;
    } else if (strcmp(argv[3], str3) == 0){
        q = INT16;
        qbits = 16;
        qsize = sizeof(int16_t);
        scale = 1<<8;
        qconv = &convolve_avx_int16;
    } else {
        printf("invalid argument\n");
        exit(-1);
    }
    float scale2 = scale * scale;
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////



    ///////////////////////////////////////////read data///////////////////////////////////////////
    // read metadata
    int isize[4];
    int ksize[4];
    fread(isize, sizeof(int), 4, ifptr);
    fread(ksize, sizeof(int), 4, kfptr);
    #ifdef DEBUG
    printf("main: precision %s\n", argv[3]);
    printf("main: (N, H, W, C) = (%d, %d, %d, %d)\n", isize[0], isize[1], isize[2], isize[3]);
    printf("main: (KH, KW, OC, IC) = (%d, %d, %d, %d)\n", ksize[0], ksize[1], ksize[2], ksize[3]);
    #endif
    N = isize[0]; H = isize[1]; W = isize[2]; C = isize[3];
    KH = ksize[0]; KW = ksize[1]; OC = ksize[2]; IC = ksize[3];
    assert(C == IC);
    #ifdef DEBUG
    printf("main: read input and kernel file into memory\n");
    #endif
    // read data
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
    fread(I, sizeof(float), N * H * W * C, ifptr);
    fread(K, sizeof(float), KH * KW * OC * IC, kfptr);
    fclose(ifptr);
    fclose(kfptr);
    ///////////////////////////////////////////read data///////////////////////////////////////////




    ///////////////////////////////////////////quantization///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: quantization bit %d\n", qbits);
    #endif
    clock_t start, end;
    start = clock();
    void * I_Q, * K_Q;
    if (qbits > 0){
        I_Q = quantize(I, q, qsize, scale, N * H * W * C);
        K_Q = quantize(K, q, qsize, scale, KH * KW * OC * IC);
    }
    else{
        I_Q = I;
        K_Q = K;
    }
    end = clock();
    float qtime = ((float) (end - start)) / CLOCKS_PER_SEC;
    ///////////////////////////////////////////quantization///////////////////////////////////////////



    ///////////////////////////////////////////main routine///////////////////////////////////////////
    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;
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
                    //if (oc==0 && h==H/2) printf("main: O[%d,%d,%d,%d]: %0.10f (restored), %0.10f (reference)\n", n, h, w, oc, O[output_idx], convolve(I, K, n, h, w, oc));
                }
            }
        }
    }
    end = clock();
    float cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("main: convolution elapsed time: %f\n", cpu_time_used);
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
    printf("main: (%s, S=%.3f) -> NRMSE=%.20f\n", argv[3], scale, NRMSE);
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
    fwrite(O, sizeof(float), N * H * W * OC, ofptr);
    fclose(ofptr);
    free(I); free(K); free(O);
    if (qbits > 0){
        free(I_Q); free(K_Q);
    }
    return 0;
    ///////////////////////////////////////////tidying up///////////////////////////////////////////
}