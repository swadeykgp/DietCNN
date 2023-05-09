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
#include "ap_int.h"
#include <iostream>
typedef float DataType;

typedef ap_uint<8> QType;

using namespace std;

static const DataType stat[] = { 47.7186, -7.2398, 99.3117, 0, 207.4083, 0, 280.4087, 0, 336.0337, 0};

// utility function for quantization

DataType calc_scale(DataType min_val, DataType max_val){
	  DataType qmin = 0;
	  DataType qmax = pow(double(2), double(nb)) - 1;
	  DataType scale = (max_val - min_val) / (qmax - qmin);
	  return scale;
}

DataType calc_zero_point(DataType min_val, DataType max_val, DataType scale){
	  DataType qmin = 0;
	  DataType qmax = pow(double(2), double(nb)) - 1;
	  DataType initial_zero_point = qmin - min_val / scale;
	  DataType zero_point = 0;

	  if (initial_zero_point < qmin)
	      zero_point = qmin;
	  else if (initial_zero_point > qmax)
	      zero_point = qmax;
	  else
	      zero_point = initial_zero_point;

	  zero_point = DataType(zero_point);
	  return zero_point;
}

void conv2D_c1(const QType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1], DataType scale_x[1], DataType zp_x[1]) {



	// Create local filter
	// This one is full implementation with channel
	const DataType M[FILTER_OUT_CHANNEL_C1][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL] = {
																					 #include "c1f.txt"
				 };
	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
		for(int j = 0; j < FILTER_SIZE_C1; j++){
			for(int k = 0; k < FILTER_SIZE_C1; k++){
				for(int l = 0; l < INP_IMAGE_CHANNEL; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
		for(int j = 0; j < FILTER_SIZE_C1; j++){
			for(int k = 0; k < FILTER_SIZE_C1; k++){
				for(int l = 0; l < INP_IMAGE_CHANNEL; l++){
					if (max_val < M[i][j][k][l])
						max_val = M[i][j][k][l];
				}
			}
		}
	}
	DataType qmin,qmax;
	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_w, zp_w;
	scale_w = calc_scale(min_val, max_val);
	zp_w = calc_zero_point(min_val, max_val, scale_w);
	QType MQ[FILTER_OUT_CHANNEL_C1][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
		for(int j = 0; j < FILTER_SIZE_C1; j++){
			for(int k = 0; k < FILTER_SIZE_C1; k++){
				for(int l = 0; l < INP_IMAGE_CHANNEL; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}

	// Modify and quantize the biases
	// Find min value
	// Create local bias
	const DataType B[FILTER_OUT_CHANNEL_C1] = {
			#include "c1b.txt"
			};
	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[FILTER_OUT_CHANNEL_C1];

	for(int i = 0; i < FILTER_OUT_CHANNEL_C1; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}





	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C1; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - FILTER_SIZE_C1; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - FILTER_SIZE_C1; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C1; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C1; n++){ // kernel columns
                                                
                            // actual multiply and add
                            tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }

    // fill the scale variables
    scale_x[0] = scale_next;
    zp_x[0] = zero_point_next;


}


void relu_c1(QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C1; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns
                   
            	P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
            }
        }
    }
}


void conv2D_c2(QType N[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_IN_CHANNEL_C2], QType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2], DataType scale_x[1], DataType zp_x[1]) {

	// Create local filter
	const DataType M[FILTER_OUT_CHANNEL_C2][FILTER_SIZE_C2][FILTER_SIZE_C2][FILTER_IN_CHANNEL_C2] = {
			#include "c2f.txt"
			};

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
		for(int j = 0; j < FILTER_SIZE_C2; j++){
			for(int k = 0; k < FILTER_SIZE_C2; k++){
				for(int l = 0; l < FILTER_IN_CHANNEL_C2; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
			for(int j = 0; j < FILTER_SIZE_C2; j++){
				for(int k = 0; k < FILTER_SIZE_C2; k++){
					for(int l = 0; l < FILTER_IN_CHANNEL_C2; l++){
					if (max_val < M[i][j][k][l])
						max_val = M[i][j][k][l];
				}
			}
		}
	}
	DataType qmin,qmax;
	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_w, zp_w;
	scale_w = calc_scale(min_val, max_val);
	zp_w = calc_zero_point(min_val, max_val, scale_w);
	QType MQ[FILTER_OUT_CHANNEL_C2][FILTER_SIZE_C2][FILTER_SIZE_C2][FILTER_IN_CHANNEL_C2];
	DataType mqpoint;

	// Process X the input - remember this is CONV 1 - use stats of CONV 2
    DataType scale_next = calc_scale(stat[5], stat[4]);
	DataType zero_point_next = calc_zero_point(stat[5], stat[4], scale_next);


	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
		for(int j = 0; j < FILTER_SIZE_C2; j++){
			for(int k = 0; k < FILTER_SIZE_C2; k++){
				for(int l = 0; l < FILTER_IN_CHANNEL_C2; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}

	// Modify and quantize the biases
	// Find min value
	// Create local bias
	const DataType B[FILTER_OUT_CHANNEL_C2] = {
			#include "c2b.txt"
			};
	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[FILTER_OUT_CHANNEL_C2];


	for(int i = 0; i < FILTER_OUT_CHANNEL_C2; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}



	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int o = 0; o < FILTER_OUT_CHANNEL_C2; o++){ // output filter
        
        //for (int i = 0; i < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; i=i+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
        for (int i = 0; i < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; i++){              // rows
        
            //for (int j = 0; j < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; j=j+STRIDE_C2){// this is correct but VISIT is having problem with loop flattening
            for (int j = 0; j < OUT_IMAGE_SIZE_P1 - FILTER_SIZE_C2; j++){          // columns
                   
                for (int l = 0; l < FILTER_IN_CHANNEL_C2; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C2; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C2; n++){ // kernel columns
                                                
                            // actual multiply and add
                        	tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C2)][(j/STRIDE_C2)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0;
            }
    
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
    scale_x[0] = scale_next;
    zp_x[0] = zero_point_next;
}


void relu_c2(QType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2]) {

	int ii = 0;
	for (int k = 0; k < FILTER_OUT_CHANNEL_C2; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows
        
            for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns
                   
            	P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
                   
            }
        }
    }
}


void maxpool_1(QType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1], QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_OUT_CHANNEL_C1]) {


	QType tmp = 0;

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
                tmp = 0;
            }
        } // end of one output channel
    }
}






void maxpool_2(QType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2], QType P[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][FILTER_OUT_CHANNEL_C2]) {


DataType tmp = 0;

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
                tmp = 0;
            }
    
        } // end of one output channel
    }
}


void fc1(QType N[OUT_IMAGE_SIZE_F1_IN], QType P[OUT_IMAGE_SIZE_F1_OUT], DataType scale_x[1], DataType zp_x[1]) {
//120 400

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN] = {
		#include "f1f.txt"
		};
    /////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){
			if (min_val > M[i][j])
						min_val = M[i][j];
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){
			if (max_val < M[i][j])
				max_val = M[i][j];
		}
	}

	DataType qmin,qmax;
	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_w, zp_w;
	scale_w = calc_scale(min_val, max_val);
	zp_w = calc_zero_point(min_val, max_val, scale_w);
	QType MQ[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[7], stat[6]);
	DataType zero_point_next = calc_zero_point(stat[7], stat[6], scale_next);
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){
			mqpoint = zp_w + M[i][j] / scale_w;
			mqpoint = mqpoint < qmin ? qmin : mqpoint;
			mqpoint = mqpoint > qmax ? qmax : mqpoint;
			mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
			MQ[i][j] = QType(mqpoint);
		}
	}

	const DataType B[OUT_IMAGE_SIZE_F1_OUT] = {
			#include "f1b.txt"
			};
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[OUT_IMAGE_SIZE_F1_OUT];

	// Process X the input - remember this is FC 1 - use stats of FC 2
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}
	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
        tmp = 0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){ //  image columns
            // actual multiply and add
            tmp +=  QType((N[j]- zp_x[0]) * M[i][j]);
        }               
        P[i] = QType(tmp + BQ[i] + zero_point_next);
        tmp = 0;
    }
    // fill the scale variables
	scale_x[0] = scale_next;
	zp_x[0] = zero_point_next;
}

void relu_fc1(QType P[OUT_IMAGE_SIZE_F1_OUT]) {


        for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
        	P[i] = ((QType)0 < (QType)P[i]) ? (QType)P[i] : (QType)0;
        }
    
}


void fc2(QType N[OUT_IMAGE_SIZE_F2_IN], QType P[OUT_IMAGE_SIZE_F2_OUT], DataType scale_x[1], DataType zp_x[1]) {
	//84 120

	// Create local filter
	const DataType M[OUT_IMAGE_SIZE_F2_OUT][OUT_IMAGE_SIZE_F2_IN] = {
			#include "f2f.txt"
			};

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F2_IN; j++){
			if (min_val > M[i][j])
						min_val = M[i][j];
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F2_IN; j++){
			if (max_val < M[i][j])
				max_val = M[i][j];
		}
	}

	DataType qmin,qmax;
	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_w, zp_w;
	scale_w = calc_scale(min_val, max_val);
	zp_w = calc_zero_point(min_val, max_val, scale_w);
	QType MQ[OUT_IMAGE_SIZE_F2_OUT][OUT_IMAGE_SIZE_F2_IN];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[9], stat[8]);
	DataType zero_point_next = calc_zero_point(stat[9], stat[8], scale_next);
	for(int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
		for(int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){
			mqpoint = zp_w + M[i][j] / scale_w;
			mqpoint = mqpoint < qmin ? qmin : mqpoint;
			mqpoint = mqpoint > qmax ? qmax : mqpoint;
			mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
			MQ[i][j] = QType(mqpoint);
		}
	}

	const DataType B[OUT_IMAGE_SIZE_F2_OUT] = {
			#include "f2b.txt"
			};
	for(int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[OUT_IMAGE_SIZE_F2_OUT];

	// Process X the input - remember this is FC 1 - use stats of FC 2
	for(int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}
	/////////////////////////////// END QUATIZATION ///////////////////////////////


    QType tmp = 0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
                        
        tmp = 0;
        for (int j = 0; j < OUT_IMAGE_SIZE_F2_IN; j++){ //  image columns
        	// actual multiply and add
			tmp +=  QType((N[j]- zp_x[0]) * M[i][j]);
		}
		P[i] = QType(tmp + BQ[i] + zero_point_next);
		tmp = 0;
    }   
    // fill the scale variables
    scale_x[0] = scale_next;
    zp_x[0] = zero_point_next;
}

void relu_fc2(QType P[OUT_IMAGE_SIZE_F2_OUT]) {


	for (int i = 0; i < OUT_IMAGE_SIZE_F2_OUT; i++){     // image rows
		P[i] = ((QType)0 < (QType)P[i]) ? (QType)P[i] : (QType)0;
	}
    
}


void fc3(QType N[OUT_IMAGE_SIZE_F3_IN], DataType P[OUT_IMAGE_SIZE_F3_OUT]) {
//10 84

	// Create local filter
	const DataType M[OUT_IMAGE_SIZE_F3_OUT][OUT_IMAGE_SIZE_F3_IN] = {
			#include "f3f.txt"
			};

	const DataType B[OUT_IMAGE_SIZE_F3_OUT] = {
		#include "f3b.txt"
		};

	DataType tmp = 0;

	for (int i = 0; i < OUT_IMAGE_SIZE_F3_OUT; i++){     // image rows
		tmp = 0;
		for (int j = 0; j < OUT_IMAGE_SIZE_F3_IN; j++){ //  image columns
			// actual multiply and add
			tmp += N[j] * M[i][j];
		}
		P[i] = tmp + B[i];
		tmp = 0;
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

	// Forward is dynamically quantized - engineer the input
	DataType qmin,qmax;
	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_x[1], zp_x[1];
	// This is the min and max form the stats of CONV 1 layer
	scale_x[0] = calc_scale(stat[1], stat[0]);
	zp_x[0] = calc_zero_point(stat[1], stat[0],  scale_x[0]);
	QType INP[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL];
	DataType mqpoint;
	for(int i = 0; i < INP_IMAGE_SIZE; i++){
		for(int j = 0; j < INP_IMAGE_SIZE; j++){
			for(int k = 0; k < INP_IMAGE_CHANNEL; k++){
				mqpoint = zp_x[0] + N_c1[i][j][k] / scale_x[0];
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				INP[i][j][k] = QType(mqpoint);
			}
		}
	}

	// do first conv2D, return the buffer for next op
	QType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][FILTER_OUT_CHANNEL_C1];
	conv2D_c1(INP, local_relu_1, scale_x, zp_x);
    relu_c1(local_relu_1);
    QType local_pool_1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][FILTER_OUT_CHANNEL_C1];
    maxpool_1(local_relu_1, local_pool_1);
    QType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][FILTER_OUT_CHANNEL_C2];
    conv2D_c2(local_pool_1, local_conv_2, scale_x, zp_x);
    relu_c2(local_conv_2);
    QType local_pool_2[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][FILTER_OUT_CHANNEL_C2];
    maxpool_2(local_conv_2, local_pool_2);
    
    QType local_fc_1[OUT_IMAGE_SIZE_F1_IN];
    
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
    QType local_fc_2[OUT_IMAGE_SIZE_F1_OUT];
    QType local_fc_3[OUT_IMAGE_SIZE_F3_IN];
    DataType local_fc_3_out[OUT_IMAGE_SIZE_F3_OUT];
    
    DataType local_softmax_out[OUT_IMAGE_SIZE_F3_OUT];
    
    fc1(local_fc_1, local_fc_2, scale_x, zp_x);
    relu_fc1(local_fc_2);

    fc2(local_fc_2, local_fc_3, scale_x, zp_x);
    relu_fc2(local_fc_3);
    
    fc3(local_fc_3, local_fc_3_out);
    
    softmax(local_fc_3_out, local_softmax_out);
    
    // finally copy this this to output
    for (int k = 0; k < OUT_SOFTMAX; k++){
    	Ps[k] = local_softmax_out[k];
    }
}
