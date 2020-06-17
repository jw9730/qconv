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
#define ALIGN_BYTES (sizeof(void *) * 2)
#define MAX_THREADS 16
typedef enum qenum{
    FP32,
    INT32,
    INT16
} DATATYPE;

//#define DEBUG
#define DO_NRMSE

FILE * ifptr, * kfptr, * ofptr;
int N, H, W, C;
int KH, KW, OC, IC;
int PH_L, PH_H, PW_L, PW_H;

struct t_arg{
    void * I_Q;
    void * K_Q;
    void * O_Q;
    float (* qconvfp) (void *, void *, int, int, int, int);
    int64_t (* qconvint) (void *, void *, int, int, int, int);
    int offset;
    int num_pixels;
    float scale2;
};

void conv_func(void * aux){
    struct t_arg * t_arg = (struct t_arg *) aux;
    void * I_Q = t_arg->I_Q;
    void * K_Q = t_arg->K_Q;
    void * O_Q = t_arg->O_Q;
    int offset = t_arg->offset;
    int num_pixels = t_arg->num_pixels;
    float scale2 = t_arg->scale2;
    // parse input pixels and perform convolution
    int c1 = (H * W * OC);
    int c2 = (W * OC);
    int c3 = (OC);
    if (t_arg->qconvfp != NULL){
        float (* qconv) (void *, void *, int, int, int, int) = t_arg->qconvfp;
        float * O = (float *) O_Q;
        for (int i=0; i<num_pixels; i++){
            int idx = offset + i;
            int n = idx/c1;
            int h = idx%c1/c2;
            int w = idx%c2/c3;
            int oc = idx%c3;
            assert(idx == INDEX_ROW_MAJOR_4(n, h, w, oc, N, H, W, OC));
            //printf("%d\t-> %d/%d, %d/%d, %d/%d, %d/%d\n", idx, n, N-1, h, H-1, w, W-1, oc, OC-1);
            O[idx] = qconv(I_Q, K_Q, n, h, w, oc);
        }
    }
    else {
        int64_t (* qconv) (void *, void *, int, int, int, int) = t_arg->qconvint;
        int64_t * O = (int64_t *) O_Q;
        for (int i=0; i<num_pixels; i++){
            int idx = offset + i;
            int n = idx/c1;
            int h = idx%c1/c2;
            int w = idx%c2/c3;
            int oc = idx%c3;
            assert(idx == INDEX_ROW_MAJOR_4(n, h, w, oc, N, H, W, OC));
            //printf("%d\t-> %d/%d, %d/%d, %d/%d, %d/%d\n", idx, n, N-1, h, H-1, w, W-1, oc, OC-1);
            O[idx] = qconv(I_Q, K_Q, n, h, w, oc);
        }
    }

}

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
void quantize_restore(float * O, void * O_Q, int size, float scale2){
    int64_t * O_int = (int64_t *) O_Q;
    for (int i=0; i<size; i++) O[i] = ((float) O_int[i]) / scale2;
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
                __m256 vx = _mm256_loadu_ps(I_p);
                __m256 vy = _mm256_loadu_ps(K_p);
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
int64_t convolve_avx_int32(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    int32_t * I = (int32_t *) I_Q;
    int32_t * K = (int32_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int64_t ret = 0;
    // parallelize multiplication with avx
    // I: row major, (N, H, W, IC)
    // K: row major, (KH, KW, OC, IC)
    // O: row major, (N, H, W, OC)
    // avx vectorization dimension: IC
    __m256i acc = (__m256i) _mm256_setzero_ps();
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
            int ic = 0;
            int residue = IC;
            int32_t * I_p = I + INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
            int32_t * K_p = K + INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
            for (int chunk=0; chunk<IC/8; chunk++){
                __m256i vx = _mm256_loadu_si256((__m256i *)I_p);
                __m256i vy = _mm256_loadu_si256((__m256i *)K_p);
                // expand to 64-bits (for precision)
                __m256i xl = _mm256_cvtepi32_epi64(_mm_loadu_si128((__m128i *)&vx));
                __m256i xh = _mm256_cvtepi32_epi64(_mm_loadu_si128(((__m128i *)&vx)+1));
                __m256i yl = _mm256_cvtepi32_epi64(_mm_loadu_si128((__m128i *)&vy));
                __m256i yh = _mm256_cvtepi32_epi64(_mm_loadu_si128(((__m128i *)&vy)+1));
                // compute product
                acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_mul_epi32(xl, yl), _mm256_mul_epi32(xh, yh)));
                ic += 8; residue -= 8; I_p += 8; K_p += 8;
            }
            // handle boundary
            __m256i vx = (__m256i) _mm256_setzero_ps();
            __m256i vy = (__m256i) _mm256_setzero_ps();
            memcpy(&vx, I_p, sizeof(int32_t) * residue);
            memcpy(&vy, K_p, sizeof(int32_t) * residue);
            // expand to 64-bits (for precision)
            __m256i xl = _mm256_cvtepi32_epi64(_mm_loadu_si128((__m128i *)&vx));
            __m256i xh = _mm256_cvtepi32_epi64(_mm_loadu_si128(((__m128i *)&vx)+1));
            __m256i yl = _mm256_cvtepi32_epi64(_mm_loadu_si128((__m128i *)&vy));
            __m256i yh = _mm256_cvtepi32_epi64(_mm_loadu_si128(((__m128i *)&vy)+1));
            // compute product
            acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_mul_epi32(xl, yl), _mm256_mul_epi32(xh, yh)));
        }
    }
    for (int k=0; k<4; k++) ret += ((int64_t *)&acc)[k];
    return ret;
}
int64_t convolve_avx_int16(void * I_Q, void * K_Q, int n, int h, int w, int oc){
    int16_t * I = (int16_t *) I_Q;
    int16_t * K = (int16_t *) K_Q;
    int IH_L = h - PH_L;
    int IW_L = w - PW_L;
    int32_t ret = 0;
    // parallelize multiplication with avx
    // I: row major, (N, H, W, IC)
    // K: row major, (KH, KW, OC, IC)
    // O: row major, (N, H, W, OC)
    // avx vectorization dimension: IC
    __m256i acc = (__m256i)_mm256_setzero_ps();
    for (int kh=0; kh<KH; kh++){
        for (int kw=0; kw<KW; kw++){
            if (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W) continue;
            int ic = 0;
            int residue = IC;
            int16_t * I_p = I + INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, C);
            int16_t * K_p = K + INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
            for (int chunk=0; chunk<IC/16; chunk++){
                __m256i vx = _mm256_loadu_si256((__m256i *)I_p);
                __m256i vy = _mm256_loadu_si256((__m256i *)K_p);
                // expand to 32-bits (for precision)
                __m256i xl = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)&vx));
                __m256i xh = _mm256_cvtepi16_epi32(_mm_loadu_si128(((__m128i *)&vx)+1));
                __m256i yl = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)&vy));
                __m256i yh = _mm256_cvtepi16_epi32(_mm_loadu_si128(((__m128i *)&vy)+1));
                // compute product
                acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_mullo_epi32(xl, yl), _mm256_mullo_epi32(xh, yh)));
                ic += 16; residue -= 16; I_p += 16; K_p += 16;
            }
            // handle boundary
            __m256i vx = (__m256i) _mm256_setzero_ps();
            __m256i vy = (__m256i) _mm256_setzero_ps();
            memcpy(&vx, I_p, sizeof(int16_t) * residue);
            memcpy(&vy, K_p, sizeof(int16_t) * residue);
            // expand to 32-bits (for precision)
            __m256i xl = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)&vx));
            __m256i xh = _mm256_cvtepi16_epi32(_mm_loadu_si128(((__m128i *)&vx)+1));
            __m256i yl = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i *)&vy));
            __m256i yh = _mm256_cvtepi16_epi32(_mm_loadu_si128(((__m128i *)&vy)+1));
            // compute product
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(_mm256_mullo_epi32(xl, yl), _mm256_mullo_epi32(xh, yh)));
        }
    }
    for (int k=0; k<8; k++) ret += ((int32_t *)&acc)[k];
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
    float (* qconvfp) (void *, void *, int, int, int, int) = NULL;
    int64_t (* qconvint) (void *, void *, int, int, int, int) = NULL;
    enum qenum q;
    int qbits = 0;
    int qsize = 0;
    float iscale, kscale;
    if (strcmp(argv[3], str1) == 0){
        q = FP32;
        qbits = 0;
        qsize = sizeof(float);
        qconvfp = &convolve_avx_fp32;
        iscale = 1;
        kscale = 1;
    } else if (strcmp(argv[3], str2) == 0){
        q = INT32;
        qbits = 32;
        qsize = sizeof(int32_t);
        iscale = (1<<9);
        kscale = (1<<9) * 5e2;
        qconvint = &convolve_avx_int32;
    } else if (strcmp(argv[3], str3) == 0){
        q = INT16;
        qbits = 16;
        qsize = sizeof(int16_t);
        iscale = (1<<7);
        kscale = (1<<7) * 5e2;
        qconvint = &convolve_avx_int16;
    } else {
        printf("invalid argument\n");
        exit(-1);
    }
    float scale2 = iscale * kscale;
    ///////////////////////////////////////////parse cmdline///////////////////////////////////////////



    ///////////////////////////////////////////read data///////////////////////////////////////////
    // read metadata
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
    ///////////////////////////////////////////read data///////////////////////////////////////////



    ///////////////////////////////////////////quantization///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: quantization bit %d\n", qbits);
    #endif
    clock_t start, end;
    start = clock();
    void * I_Q, * K_Q, * O_Q;
    if (qbits > 0){
        I_Q = quantize(I, q, qsize, iscale, N * H * W * C);
        K_Q = quantize(K, q, qsize, kscale, KH * KW * OC * IC);
        if ((rc = posix_memalign((void **)&O_Q, ALIGN_BYTES, N * H * W * OC * sizeof(int64_t))) != 0){
            printf("main: output memory allocation failure\n");
            exit(-1);
        }
    } else if (qbits == 0){
        I_Q = I;
        K_Q = K;
        if ((rc = posix_memalign((void **)&O_Q, ALIGN_BYTES, N * H * W * OC * sizeof(float))) != 0){
            printf("main: output memory allocation failure\n");
            exit(-1);
        }
    } else {
        printf("main: invalid quantization option\n");
        exit(-1);
    }
    end = clock();
    float qtime = ((float) (end - start)) / CLOCKS_PER_SEC;
    ///////////////////////////////////////////quantization///////////////////////////////////////////



    ///////////////////////////////////////////setup threading///////////////////////////////////////////
    int pix_per_thread = (N*H*W*OC + MAX_THREADS - 1) / MAX_THREADS;
    int num_thread = N*H*W*OC/pix_per_thread;
    pthread_t tid[MAX_THREADS];
    struct t_arg t_args[MAX_THREADS];
    #ifdef DEBUG
    printf("main: number of pixels %d, %d threads each with %d pixels will run\n", N*H*W*OC, num_thread, pix_per_thread);
    #endif
    ///////////////////////////////////////////setup threading///////////////////////////////////////////



    ///////////////////////////////////////////main routine///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: compute convolution into output @ %p\n", O);
    #endif
    start = clock();
    int residue = N*H*W*OC;
    struct t_arg * t_arg = t_args;
    int t = 0;
    while(residue>0){
        t_arg->I_Q = I_Q;
        t_arg->K_Q = K_Q;
        t_arg->O_Q = O_Q;
        t_arg->qconvfp = qconvfp;
        t_arg->qconvint = qconvint;
        t_arg->offset = pix_per_thread * t;
        t_arg->num_pixels = (residue < pix_per_thread) ? residue : pix_per_thread;
        t_arg->scale2 = scale2;
        pthread_create(tid + t, NULL, conv_func, t_arg);
        residue -= pix_per_thread; t++; t_arg++;
    }
    for (int i=0; i<t; i++){
        pthread_join(tid[i], NULL);
    }
    end = clock();
    float ctime = ((float) (end - start)) / CLOCKS_PER_SEC;
    ///////////////////////////////////////////main routine///////////////////////////////////////////



    ///////////////////////////////////////////postprocessing///////////////////////////////////////////
    start = clock();
    if (qbits == 0) memcpy(O, O_Q, N*H*W*OC*sizeof(float));
    else quantize_restore(O, O_Q, N*H*W*OC, scale2);
    end = clock();
    qtime += ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("%s convolution %fs, overhead %fs\n", argv[3], ctime, qtime);
    ///////////////////////////////////////////postprocessing///////////////////////////////////////////



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
    if (qbits > 0){
        free(I_Q); free(K_Q);
    }
    return 0;
    ///////////////////////////////////////////tidying up///////////////////////////////////////////
}