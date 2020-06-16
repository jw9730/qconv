#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

#define DEBUG
#define INDEX_ROW_MAJOR_2(i, j, I, J) ((j) + (J) * (i))
#define INDEX_ROW_MAJOR_3(i, j, k, I, J, K) ((k) + (K) * ((j) + (J) * (i)))
#define INDEX_ROW_MAJOR_4(i, j, k, l, I, J, K, L) ((l) + (L) * ((k) + (K) * ((j) + (J) * (i))))

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
    int flag;
    for (int ic=0; ic<IC; ic++){
        for (int kh=0; kh<KH; kh++){
            for (int kw=0; kw<KW; kw++){
                flag = (IH_L+kh < 0 || IH_L+kh >= H || IW_L+kw < 0 || IW_L+kw >= W);
                if (flag){
                    printf("\t+= I[%d,%d/%d,%d/%d,%d] * K[%d,%d,%d,%d], out-of-bounds: %d \n", n, IH_L+kh, H-1, IW_L+kw, W-1, ic, kh, kw, oc, ic, flag);
                    continue;
                }
                int input_idx = INDEX_ROW_MAJOR_4(n, IH_L+kh, IW_L+kw, ic, N, H, W, OC);
                int kernel_idx = INDEX_ROW_MAJOR_4(kh, kw, oc, ic, KH, KW, OC, IC);
                ret += I[input_idx] * K[kernel_idx];
            }
        }
    }
    return ret;
}

int main(int argc, char **argv){
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

    // compute padding (TensorFlow pads more on higher index)
    PH_H = (KH + 1)/2;
    PH_L = KH - PH_H;
    PW_H = (KW + 1)/2;
    PW_L = KW - PW_H;

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
                    printf("main: O[%d,%d,%d,%d] = ?\n", n, h, w, oc, O[output_idx]);
                    O[output_idx] = convolve(I, K, n, h, w, oc);
                    printf("main: O[%d,%d,%d,%d] = %0.3f\n", n, h, w, oc, O[output_idx]);
                }
            }
        }
    }

    end = clock();
    float cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("main: convolution elapsed time: %f\n", cpu_time_used);

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
    return 0;
}