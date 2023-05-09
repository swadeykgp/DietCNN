#ifndef CONV_H
#define CONV_H




#define INP_IMAGE_SIZE 32
#define INP_IMAGE_CHANNEL 3
#define OUT_IMAGE_SIZE_C1 15
#define OUT_IMAGE_SIZE_C2 13
#define OUT_IMAGE_SIZE_C3 11
#define OUT_IMAGE_SIZE_C4 9
#define OUT_IMAGE_SIZE_C5 7
#define OUT_IMAGE_SIZE_C6 5
#define OUT_IMAGE_SIZE_C7 3
#define OUT_IMAGE_SIZE_C8 1

#define OUT_IMAGE_SIZE_F1_IN  512
#define OUT_IMAGE_SIZE_F1_OUT  10

#define FILTER_SIZE 3
#define STRIDE_C1 2
#define STRIDE_ALL 1

#define N_CLUSTERS 512
#define N_CFILTERS 256
#define N_FFILTERS 128
#define SYM_STRIDE 1
//#define PATCH_SIZE (1, 1)
#define MAX_WH 32
#define MAX_CH 512
#define MAX_MEM 524288 //32*32*512

#define  C1B  64
#define  C2B  128
#define  C3B  256
#define  C4B  256
#define  C5B  512
#define  C6B  512
#define  C7B  512
#define  C8B  512

#define  F1B  512
#endif
