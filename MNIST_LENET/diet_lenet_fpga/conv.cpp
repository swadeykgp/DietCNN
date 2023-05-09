//#include <sys/types.h>
//#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>
#include "conv.h"


typedef int DataType;
typedef float ReturnType;

using namespace std;

int compare_integers (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


void conv2D_c1(const DataType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[FILTER_OUT_CHANNEL_C1][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1fs.txt"
		     };
const DataType C1BL[N_CLUSTERS][FILTER_OUT_CHANNEL_C1] = {
							#include "c1b_lut.txt"
		         };
int tmp = 0;
const int tmp_symbol_num = INP_IMAGE_CHANNEL*FILTER_SIZE_C1*FILTER_SIZE_C1;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < FILTER_OUT_CHANNEL_C1; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - FILTER_SIZE_C1; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - FILTER_SIZE_C1; j=j+STRIDE_C1){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){ // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C1; m++){ // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C1; n++){ // kernel columns
                                                
                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE_C1*FILTER_SIZE_C1 + m*FILTER_SIZE_C1 + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
                        }       
                    }   
                } // end of one window , all input channels . now ripple addition will start

                // sort temporary symbol array
                //qsort(tmp_conv_sym, tmp_symbol_num, sizeof(int), compare_integers);

                // Get the first symbol
                tmp = tmp_conv_sym[0];

                // ripple add next symbol onwards
                for (int i = 1; i < tmp_symbol_num; i++){
                	tmp = AL[tmp][tmp_conv_sym[i]];
                }

                // add bias
                tmp = C1BL[tmp][o];
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}


void relu_c1(DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1], const DataType RL[N_CLUSTERS]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C1; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns
                   
            	P[i][j][k] = RL[P[i][j][k]];
            }
        }
    }
}


void conv2D_c2(DataType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_IN_CHANNEL_C2],
		DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
const DataType M[FILTER_OUT_CHANNEL_C2][FILTER_SIZE_C2][FILTER_SIZE_C2][FILTER_IN_CHANNEL_C2] = {
		#include "c2fs.txt"
		};
const DataType C2BL[N_CLUSTERS][C2B] = {
								#include "c2b_lut.txt"
			     };

    int tmp = 0;
    const int tmp_symbol_num = FILTER_IN_CHANNEL_C2*FILTER_SIZE_C2*FILTER_SIZE_C2;
    DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < FILTER_OUT_CHANNEL_C2; o++){ // output filter
        
        //for (int i = 0; i < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; i=i+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE_C2; i++){              // rows
        
            //for (int j = 0; j < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; j=j+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE_C2; j++){          // columns
                   
                for (int l = 0; l < FILTER_IN_CHANNEL_C2; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C2; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C2; n++){ // kernel columns
                                                
                        	// actual multiply and add replaced by table lookup
							tmp_conv_sym[l*FILTER_SIZE_C2*FILTER_SIZE_C2 + m*FILTER_SIZE_C2 + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
						}
					}
				} // end of one window , all input channels . now ripple addition will start

				// sort temporary symbol array
				//qsort(tmp_conv_sym, tmp_symbol_num, sizeof(int), compare_integers);

				// Get the first symbol
				tmp = tmp_conv_sym[0];

				// ripple add next symbol onwards
				for (int i = 1; i < tmp_symbol_num; i++){
					tmp = AL[tmp][tmp_conv_sym[i]];
				}

				// add bias
				tmp = C2BL[tmp][o];
				P[(i/STRIDE_C2)][(j/STRIDE_C2)][o] =  tmp;
				tmp = 0;
            }
    
        }
    }
}


void relu_c2(DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2], const DataType RL[N_CLUSTERS]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C2; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns
                   
            	P[i][j][k] = RL[P[i][j][k]];
                   
            }
        }
    }
}


void fc1(DataType N[OUT_IMAGE_SIZE_F1_IN],
		DataType P[OUT_IMAGE_SIZE_F1_OUT],
		const DataType FL[N_CLUSTERS][N_FFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//120 400

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN] = {
		#include "f1fs.txt"
		};

const DataType F1BL[N_CLUSTERS][OUT_IMAGE_SIZE_F1_OUT] = {
								#include "f1b_lut.txt"
			     };
   int tmp = 0;
   const int tmp_symbol_num = OUT_IMAGE_SIZE_F1_IN;
   DataType tmp_sym[tmp_symbol_num];

    for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
                        
        tmp = 0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){ //  image columns
            // actual multiply and add replaced by table lookup
        	tmp_sym[j] = FL[N[j]][M[i][j]]; // Matrix Multiplication
        }
        // end of one window , all input channels . now ripple addition will start

		// sort temporary symbol array
		//qsort(tmp_sym, tmp_symbol_num, sizeof(int), compare_integers);

		// Get the first symbol
		tmp = tmp_sym[0];

		// ripple add next symbol onwards
		for (int i = 1; i < tmp_symbol_num; i++){
			tmp = AL[tmp][tmp_sym[i]];
		}

		// add bias
		tmp = F1BL[tmp][i];
		P[i] =  tmp;
		tmp = 0;
    }   
}

void relu_fc1(DataType P[OUT_IMAGE_SIZE_F1_OUT], const DataType RL[N_CLUSTERS]) {


        for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
            P[i] = RL[P[i]];
        }
    
}


void fc2(DataType N[OUT_IMAGE_SIZE_F2_IN],
		DataType P[OUT_IMAGE_SIZE_F2_OUT],
		const DataType FL[N_CLUSTERS][N_FFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//84 120

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F2_OUT][OUT_IMAGE_SIZE_F2_IN] = {
		#include "f2fs.txt"
		};
const DataType F2BL[N_CLUSTERS][OUT_IMAGE_SIZE_F2_OUT] = {
								#include "f2b_lut.txt"
			     };
int tmp =0;
const int tmp_symbol_num = OUT_IMAGE_SIZE_F2_IN;
DataType tmp_sym[tmp_symbol_num];
    for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
                        
        tmp = 0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F2_IN; j++){ //  image columns
        	// actual multiply and add replaced by table lookup
        	tmp_sym[j] = FL[N[j]][M[i][j]]; // convolution
        }               
        // end of one window , all input channels . now ripple addition will start

		// sort temporary symbol array
		//qsort(tmp_sym, tmp_symbol_num, sizeof(int), compare_integers);

		// Get the first symbol
		tmp = tmp_sym[0];

		// ripple add next symbol onwards
		for (int i = 1; i < tmp_symbol_num; i++){
			tmp = AL[tmp][tmp_sym[i]];
		}

		// add bias
		tmp = F2BL[tmp][i];
		P[i] =  tmp;
		tmp = 0;
    }   
}

void relu_fc2(DataType P[OUT_IMAGE_SIZE_F2_OUT], const DataType RL[N_CLUSTERS]) {


	for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
		 P[i] = RL[P[i]];
	}
    
}


void fc3(DataType N[OUT_IMAGE_SIZE_F3_IN],
		DataType P[OUT_IMAGE_SIZE_F3_OUT],
		const DataType FL[N_CLUSTERS][N_FFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//10 84

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F3_OUT][OUT_IMAGE_SIZE_F3_IN] = {
		#include "f3fs.txt"
		};
const DataType F3BL[N_CLUSTERS][OUT_IMAGE_SIZE_F1_OUT] = {
								#include "f3b_lut.txt"
        };

int tmp =0;
const int tmp_symbol_num = OUT_IMAGE_SIZE_F3_IN;
DataType tmp_sym[tmp_symbol_num];

    for (int i = 0; i < OUT_IMAGE_SIZE_F3_OUT; i++){     // image rows
                        
        for (int j = 0; j < OUT_IMAGE_SIZE_F3_IN; j++){ //  image columns
        	// actual multiply and add replaced by table lookup
			tmp_sym[j] = FL[N[j]][M[i][j]]; // convolution
		}
		// end of one window , all input channels . now ripple addition will start

		// sort temporary symbol array
		//qsort(tmp_sym, tmp_symbol_num, sizeof(int), compare_integers);

		// Get the first symbol
		tmp = tmp_sym[0];

		// ripple add next symbol onwards
		for (int i = 1; i < tmp_symbol_num; i++){
			tmp = AL[tmp][tmp_sym[i]];
		}

		// add bias
		tmp = F3BL[tmp][i];
		P[i] =  tmp;
		tmp = 0;
    }   
}

// Copied & modified with gratitude from https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c

void softmax(ReturnType N[OUT_SOFTMAX], ReturnType P[OUT_SOFTMAX] )
{
    int i;
    ReturnType m;
    /* Find maximum value from input array */
    m = N[0];
    for (i = 1; i < OUT_SOFTMAX; i++) {
        if (N[i] > m) {
            m = N[i];
        }
    }

    ReturnType sum = 0;
    for (i = 0; i < OUT_SOFTMAX; i++) {
        sum += expf(N[i]-m);
    }

    for (i = 0; i < OUT_SOFTMAX; i++) {
        P[i] = expf(N[i] - m - log(sum));

    }    
}

// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		ReturnType Ps[OUT_SOFTMAX]) {

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp
#pragma HLS INTERFACE mode=axis port=N_c1
#pragma HLS INTERFACE mode=axis port=Ps


	const DataType CL[N_CLUSTERS][N_CFILTERS] = {
			#include "conv_lut.txt"
			     };
	const DataType AL[N_CLUSTERS][N_CLUSTERS] = {
					#include "add_lut.txt"
				 };
	const DataType RL[N_CLUSTERS] = {
						#include "relu_lut.txt"
				 };
	const DataType FL[N_CLUSTERS][N_FFILTERS] = {
				#include "fc_lut.txt"
				     };

	// The centroid LUT
	const ReturnType CODEBOOK[128] = {
					#include "centroid_lut.txt"
			 };

	// do first conv2D, return the buffer for next op
	DataType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1];
	conv2D_c1(N_c1, local_relu_1, CL, AL);
    relu_c1(local_relu_1, RL);
    DataType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2];
    conv2D_c2(local_relu_1, local_conv_2, CL, AL);
    relu_c2(local_conv_2, RL);
    
    DataType local_fc_1[OUT_IMAGE_SIZE_F1_IN];
    
    // Flatten buffer
    for (int k = 0; k < FILTER_OUT_CHANNEL_C2; k++){     // image rows
        for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){     // image rows
            for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){ //  image columns
                // actual multiply and add
        	    local_fc_1[k*OUT_IMAGE_SIZE_C2*OUT_IMAGE_SIZE_C2 + i*OUT_IMAGE_SIZE_C2 + j] = local_conv_2[i][j][k];
            }
        }
    }
    
    // Fully connected ones
    DataType local_fc_2[OUT_IMAGE_SIZE_F1_OUT];
    DataType local_fc_3[OUT_IMAGE_SIZE_F3_IN];
    DataType local_fc_3_out[OUT_IMAGE_SIZE_F3_OUT];
    
    ReturnType local_fc_3_out_decoded[OUT_IMAGE_SIZE_F3_OUT];
    ReturnType local_softmax_out[OUT_IMAGE_SIZE_F3_OUT];
    
    fc1(local_fc_1, local_fc_2, FL, AL);
    relu_fc1(local_fc_2, RL);

    fc2(local_fc_2, local_fc_3, FL, AL);
    relu_fc2(local_fc_3, RL);
    
    fc3(local_fc_3, local_fc_3_out, FL, AL);
    
    for (int k = 0; k < OUT_IMAGE_SIZE_F3_OUT; k++){
    	local_fc_3_out_decoded[k] = CODEBOOK[local_fc_3_out[k]];
        }


    softmax(local_fc_3_out_decoded, local_softmax_out);
    
    // finally copy this this to output
    for (int k = 0; k < OUT_SOFTMAX; k++){
    	Ps[k] = local_softmax_out[k];
    }
}
