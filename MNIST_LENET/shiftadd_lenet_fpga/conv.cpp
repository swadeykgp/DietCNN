//#include <sys/types.h>
//#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>
#include "conv.h"


typedef float DataType;

using namespace std;


void conv2D_c1(const DataType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[FILTER_OUT_CHANNEL_C1][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };
// Create local bias
const DataType B[FILTER_OUT_CHANNEL_C1] = {
		#include "c1b.txt"
		};

DataType tmp = 0.0;
int tmp_shift1 = 0.0;
int tmp_shift2 = 0.0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C1; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - FILTER_SIZE_C1; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - FILTER_SIZE_C1; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C1; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C1; n++){ // kernel columns


                        	// Original implementation has following logic for shift based on the signs
                        	// to keep the essence we use a single shift for the resource estimation
	//                        	if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
	//								y -= (x << s);
	//							}
	//							else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
	//								y += (x << s);
	//							}
	//							else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
	//								y -= (x >> (-s));
	//							}
	//							else{
	//								y += (x >> (-s));
	//							}
                        	tmp_shift1 = (int)N[i+m][j+n][l];
							tmp_shift2 = (int)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);


                                                
                            // actual L1 Norm , sum of absolute difference between points in filter
                            tmp += abs(N[i+m][j+n][l] - M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}


void relu_c1(DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C1; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns
                   
            	P[i][j][k] = (0 < P[i][j][k]) ? P[i][j][k] : 0;
            }
        }
    }
}


void conv2D_c2(DataType N[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_IN_CHANNEL_C2], DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2]) {

// Create local filter
const DataType M[FILTER_OUT_CHANNEL_C2][FILTER_SIZE_C2][FILTER_SIZE_C2][FILTER_IN_CHANNEL_C2] = {
		#include "c2f.txt"
		};
const DataType B[FILTER_OUT_CHANNEL_C2] = {
		#include "c2b.txt"
		};

DataType tmp = 0.0;
int tmp_shift1 = 0.0;
int tmp_shift2 = 0.0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C2; o++){ // output filter
        
        //for (int i = 0; i < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; i=i+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
        for (int i = 0; i < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; i++){              // rows
        
            //for (int j = 0; j < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; j=j+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
            for (int j = 0; j < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; j++){          // columns
                   
                for (int l = 0; l < FILTER_IN_CHANNEL_C2; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C2; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C2; n++){ // kernel columns
                        	// Original implementation has following logic for shift based on the signs
                        	// to keep the essence we use a single shift for the resource estimation
	//                        	if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
	//								y -= (x << s);
	//							}
	//							else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
	//								y += (x << s);
	//							}
	//							else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
	//								y -= (x >> (-s));
	//							}
	//							else{
	//								y += (x >> (-s));
	//							}
                        	tmp_shift1 = (int)N[i+m][j+n][l];
							tmp_shift2 = (int)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);
                        	// actual L1 Norm , sum of absolute difference between points in filter
                            tmp += abs(N[i+m][j+n][l] - M[o][m][n][l]);

                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C2)][(j/STRIDE_C2)][o] =  tmp + B[o];
                tmp = 0.0;
            }
    
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}


void relu_c2(DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C2; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns
                   
            	P[i][j][k] = (0 < P[i][j][k]) ? P[i][j][k] : 0;
                   
            }
        }
    }
}


void maxpool_1(DataType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1], DataType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_OUT_CHANNEL_C1]) {


DataType tmp = 0.0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C1; o++){ // image channels
        
        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE_P; i=i+STRIDE_P){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE_P; j=j+STRIDE_P){          // columns
                   
                for (int m = 0; m < FILTER_SIZE_P; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_P; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }       
                } // end of one maxpool window 
                P[(i/STRIDE_P)][(j/STRIDE_P)][o] =  tmp;
                tmp = 0.0;
            }
        } // end of one output channel
    }
}






void maxpool_2(DataType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2], DataType P[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][FILTER_OUT_CHANNEL_C2]) {


DataType tmp = 0.0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C2; o++){ // output filter
        
        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE_P; i=i+STRIDE_P){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE_P; j=j+STRIDE_P){          // columns
                   
                for (int m = 0; m < FILTER_SIZE_P; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_P; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }       
                } // end of one maxpool window 
                P[(i/STRIDE_P)][(j/STRIDE_P)][o] =  tmp;
                tmp = 0.0;
            }
    
        } // end of one output channel
    }
}


void fc1(DataType N[OUT_IMAGE_SIZE_F1_IN], DataType P[OUT_IMAGE_SIZE_F1_OUT]) {
//120 400

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN] = {
		#include "f1f.txt"
		};
const DataType B[OUT_IMAGE_SIZE_F1_OUT] = {
		#include "f1b.txt"
		};
DataType tmp = 0.0;
int tmp_shift1 = 0.0;
int tmp_shift2 = 0.0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
                        
        tmp = 0.0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){ //  image columns
        	// actual L1 Norm , sum of absolute difference between points in filter
        	// Original implementation has following logic for shift based on the signs
        	// to keep the essence we use a single shift for the resource estimation
//                        	if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
//								y -= (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
//								y += (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
//								y -= (x >> (-s));
//							}
//							else{
//								y += (x >> (-s));
//							}
        	tmp_shift1 = (int)N[j];
			tmp_shift2 = (int)M[i][j];
			tmp += (tmp_shift1 >> tmp_shift2);
        	tmp += abs(N[j] - M[i][j]);
        }               
        P[i] = tmp;
        tmp = 0.0;
    }   
}

void relu_fc1(DataType P[OUT_IMAGE_SIZE_F1_OUT]) {


        for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
            P[i] = (0 < P[i]) ? P[i] : 0;
        }
    
}


void fc2(DataType N[OUT_IMAGE_SIZE_F2_IN], DataType P[OUT_IMAGE_SIZE_F2_OUT]) {
//84 120

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F2_OUT][OUT_IMAGE_SIZE_F2_IN] = {
		#include "f2f.txt"
		};
const DataType B[OUT_IMAGE_SIZE_F2_OUT] = {
		#include "f2b.txt"
		};
DataType tmp = 0.0;
int tmp_shift1 = 0.0;
int tmp_shift2 = 0.0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
                        
        tmp = 0.0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F2_IN; j++){ //  image columns
        	// actual L1 Norm , sum of absolute difference between points in filter
        	// Original implementation has following logic for shift based on the signs
        	// to keep the essence we use a single shift for the resource estimation
//                        	if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
//								y -= (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
//								y += (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
//								y -= (x >> (-s));
//							}
//							else{
//								y += (x >> (-s));
//							}
        	tmp_shift1 = (int)N[j];
			tmp_shift2 = (int)M[i][j];
			tmp += (tmp_shift1 >> tmp_shift2);
        	// actual L1 Norm , sum of absolute difference between points in filter
            tmp += abs(N[j] - M[i][j]);
        }               
        P[i] = tmp;
        tmp = 0.0;
    }   
}

void relu_fc2(DataType P[OUT_IMAGE_SIZE_F2_OUT]) {


	for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
	            P[i] = (0 < P[i]) ? P[i] : 0;
	}
    
}


void fc3(DataType N[OUT_IMAGE_SIZE_F3_IN], DataType P[OUT_IMAGE_SIZE_F3_OUT]) {
//10 84

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F3_OUT][OUT_IMAGE_SIZE_F3_IN] = {
		#include "f3f.txt"
		};
const DataType B[OUT_IMAGE_SIZE_F3_OUT] = {
		#include "f3b.txt"
		};
DataType tmp = 0.0;
int tmp_shift1 = 0.0;
int tmp_shift2 = 0.0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F3_OUT; i++){     // image rows
                        
        for (int j = 0; j < OUT_IMAGE_SIZE_F3_IN; j++){ //  image columns
        	// actual L1 Norm , sum of absolute difference between points in filter
        	// Original implementation has following logic for shift based on the signs
        	// to keep the essence we use a single shift for the resource estimation
//                        	if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
//								y -= (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
//								y += (x << s);
//							}
//							else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
//								y -= (x >> (-s));
//							}
//							else{
//								y += (x >> (-s));
//							}
        	tmp_shift1 = (int)N[j];
			tmp_shift2 = (int)M[i][j];
			tmp += (tmp_shift1 >> tmp_shift2);
        	// actual L1 Norm , sum of absolute difference between points in filter
            tmp += abs(N[j] - M[i][j]);
        }               
        P[i] = tmp;
        tmp = 0.0;
    }   
}

// Copied & modified with gratitude from https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c

void softmax(DataType N[OUT_SOFTMAX], DataType P[OUT_SOFTMAX] )
{
    int i;
    DataType m;
    /* Find maximum value from input array */
    m = N[0];
    for (i = 1; i < OUT_SOFTMAX; i++) {
        if (N[i] > m) {
            m = N[i];
        }
    }

    DataType sum = 0;
    for (i = 0; i < OUT_SOFTMAX; i++) {
        sum += expf(N[i]-m);
    }

    for (i = 0; i < OUT_SOFTMAX; i++) {
        P[i] = expf(N[i] - m - log(sum));

    }    
}

// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType Ps[OUT_SOFTMAX]) {

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp

#pragma HLS INTERFACE mode=axis port=N_c1
#pragma HLS INTERFACE mode=axis port=Ps

//	DataType local_input[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL];
//#pragma HLS ARRAY_PARTITION variable=local_input type=complete dim=2

//	read_input:
//		for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
//			for (int i = 0; i < INP_IMAGE_SIZE; i++){     // image rows
//				for (int j = 0; j < INP_IMAGE_SIZE; j++){ // image columns
//					// actual multiply and add
//					local_input[i][j][l] = N_c1[i][j][l];
//				}
//			}
//		}

	// do first conv2D, return the buffer for next op
	DataType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1];
	conv2D_c1(N_c1, local_relu_1);
    relu_c1(local_relu_1);
    DataType local_pool_1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_OUT_CHANNEL_C1];
    maxpool_1(local_relu_1, local_pool_1);
    DataType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2];
    conv2D_c2(local_pool_1, local_conv_2);
    relu_c2(local_conv_2);
    DataType local_pool_2[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][FILTER_OUT_CHANNEL_C2];
    maxpool_2(local_conv_2, local_pool_2);
    
    DataType local_fc_1[OUT_IMAGE_SIZE_F1_IN];
    
    // Flatten buffer
    for (int k = 0; k < FILTER_OUT_CHANNEL_C2; k++){     // image rows
        for (int i = 0; i < OUT_IMAGE_SIZE_P2; i++){     // image rows
            for (int j = 0; j < OUT_IMAGE_SIZE_P2; j++){ //  image columns
                // actual multiply and add
        	    local_fc_1[k*OUT_IMAGE_SIZE_P2*OUT_IMAGE_SIZE_P2 + i*OUT_IMAGE_SIZE_P2 + j] = local_pool_2[i][j][k];
            }
        }
    }
    
    // Fully connected ones
    DataType local_fc_2[OUT_IMAGE_SIZE_F1_OUT];
    DataType local_fc_3[OUT_IMAGE_SIZE_F3_IN];
    DataType local_fc_3_out[OUT_IMAGE_SIZE_F3_OUT];
    
    DataType local_softmax_out[OUT_IMAGE_SIZE_F3_OUT];
    
    fc1(local_fc_1, local_fc_2);
    relu_fc1(local_fc_2);

    fc2(local_fc_2, local_fc_3);
    relu_fc2(local_fc_3);
    
    fc3(local_fc_3, local_fc_3_out);
    
    softmax(local_fc_3_out, local_softmax_out);
    
    // finally copy this this to output
    for (int k = 0; k < OUT_SOFTMAX; k++){
    	Ps[k] = local_softmax_out[k];
    }
}
