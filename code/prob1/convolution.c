#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))
#define ALIGN_BYTES (sizeof(void *) * 2)

//#define DEBUG

FILE * ifptr, * kfptr, * ofptr;
int N, H, W, C;
int KH, KW, OC, IC;
int PH_L, PH_H, PW_L, PW_H;
int PH, PW;

void zero_pad(float * PI, float * I, int N, int H, int W, int C, int KH, int KW){
    memset(PI, 0, sizeof(float) * N * (H+KW) * (W+KW) * C);
    for (int n=0; n<N; n++){
        for (int ic=0; ic<C; ic++){
            for (int h=0; h<H; h++){
                for (int w=0; w<W; w++){
                    // h, w: position in padded input
                    // position in original input: subtract lower pad
                    int pi_index = INDEX_ROW_MAJOR_4(n,h+PH_L,w+PW_L,ic, N,PH,PW,C);
                    int in_index = INDEX_ROW_MAJOR_4(n,h,w,ic, N,H,W,C);
                    PI[pi_index] = I[in_index];
                }
            }
        }
    }
}
float convolve(float * PI, float * K, int n, int h, int w, int oc){
    // gets padded input and kernel array, outputs a convolved output value
    // position in padded input
    float ret = 0;
    int input_idx;
    int kernel_idx;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                input_idx = INDEX_ROW_MAJOR_4(n,h+kh,w+kw,ic, N,PH,PW,C);
                kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += PI[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}

int main(int argc, char **argv){
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
    ///////////////////////////////////////////read data///////////////////////////////////////////



    ///////////////////////////////////////////zero pad///////////////////////////////////////////
    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;
    PH = H + KH;
    PW = W + KW;
    // declared padded input array
    float * PI;
    if ((rc = posix_memalign((void **)&PI, ALIGN_BYTES, N * PH * PW * C * sizeof(float))) != 0){
        printf("main: input memory allocation failure\n");
        exit(-1);
    }
    zero_pad(PI, I, N, H, W, C, KH, KW);
    ///////////////////////////////////////////zero pad///////////////////////////////////////////



    ///////////////////////////////////////////main routine///////////////////////////////////////////
    #ifdef DEBUG
    printf("main: compute convolution into output @ %p\n", O);
    #endif
    clock_t start, end;
    start = clock();
    // compute convolution (scalar operations)
    for (int n=0; n<N; n++){
        for (int h=0; h<H; h++){
            for (int w=0; w<W; w++){
                for (int oc=0; oc<OC; oc++){
                    // convolution for a single output pixel
                    int output_idx = INDEX_ROW_MAJOR_4(n, h, w, oc, N, H, W, OC);
                    //printf("main: compute O[%d,%d,%d,%d], currently %0.3f\n", n, h, w, oc, O[output_idx]);
                    O[output_idx] = convolve(PI, K, n, h, w, oc);
                }
            }
        }
    }
    end = clock();
    float ctime = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("convolution %fs\n", ctime);
    ///////////////////////////////////////////main routine///////////////////////////////////////////



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
    free(I); free(K); free(O); free(PI);
    return 0;
    ///////////////////////////////////////////tidying up///////////////////////////////////////////
}