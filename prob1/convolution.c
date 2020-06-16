#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define DEBUG

FILE * ifile, * kfile;
size_t read_size;

int
main (int argc, char **argv)
{
    #ifdef DEBUG
    printf("main: argc=%d\n", argc);
    #endif

    assert (argc == 2);
    // open file
    if (argv[0] == NULL || argv[1] == NULL) {
        printf("./convolution [input_tensor.bin] [kernel_tensor.bin]\n");
        exit(-1);
    }
    if (ifile = fopen(argv[0], "r") == NULL){
        printf("invalid input file\n");
        exit(-1);
    }
    if (kfile = fopen(argv[1], "r") == NULL){
        printf("invalid kernel file\n");
        exit(-1);
    }
    // reading metadata
    int isize[4];
    int ksize[4];
    read_size = fread(isize, sizeof(int), 4, ifile);
    read_size = fread(ksize, sizeof(int), 4, kfile);
    printf("(N, H, W, C) = (%d, %d, %d, %d)\n", isize[0], isize[1], isize[2], isize[3]);
    printf("(KH, KW, OC, IC) = (%d, %d, %d, %d)\n", ksize[0], ksize[1], ksize[2], ksize[3]);
}