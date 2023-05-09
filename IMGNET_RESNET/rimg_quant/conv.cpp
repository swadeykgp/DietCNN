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
#include<cstdio>
#include<cstdlib>
#include "ap_int.h"
#define nb 8


typedef float DataType;

typedef ap_uint<8> QType;


using namespace std;

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


static const DataType stat[] = { 38.0929, -34.3413, 104.3430, 0, 107.0603, 0, 74.4338, 0, 52.3959, 0, 24.7309, 0, 31.2128, 0, 62.8690, 0};

int compare_integers (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}



// For main forward

void conv2D_c1(const QType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };

/////////////////////////////// START QUATIZATION ///////////////////////////////

// Modify and quantize the bias
// Find min value
DataType min_val;
for(int i = 0; i < C1B; i++){
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
for(int i = 0; i < C1B; i++){
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
QType MQ[C1B][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL];
DataType mqpoint;
// Process X the input - remember this is CONV 1 - use stats of CONV 2
DataType scale_next = calc_scale(stat[3], stat[2]);
DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

for(int i = 0; i < C1B; i++){
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



QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - FILTER_SIZE_C1; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - FILTER_SIZE_C1; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE_C1; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE_C1; n++){ // kernel columns
                                                
                            // actual multiply and add
                            tmp += (QType)(N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}
void ReLU1(QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void maxpool_1(QType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B]) {


	QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_ALL; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel
    }
}

// For layer 1

void conv2D_layer1(QType N[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B],
	QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B], const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][C1B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_P1_PAD][OUT_IMAGE_SIZE_P1_PAD][C1B];
	for (int i = 0; i < C1B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_P1; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_P1; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C1B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C1B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
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
	QType MQ[C1B][FILTER_SIZE][FILTER_SIZE][C1B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C1B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}





	QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_P1 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_P1 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}

void ReLU_layer1(QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_P1; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_P1; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void maxpool_layer1(QType N[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B], QType P[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C1B]) {


	QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_P1 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_P1 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_ALL; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel
    }
}


// For Layer 2
void conv2D_layer2_first(QType N[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C1B],
		QType P[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B], const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B], DataType scale_x[1], DataType zp_x[1]) {
	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Find min value
	DataType min_val;
	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
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
	QType MQ[C2B][FILTER_SIZE][FILTER_SIZE][C1B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C1B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}


	QType tmp = 0;

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL1 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL1 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}




void conv2D_layer2(QType N[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B],
		QType P[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B], const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C2B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_PL1_PAD][OUT_IMAGE_SIZE_PL1_PAD][C2B];
	for (int i = 0; i < C1B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_PL1; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_PL1; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
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
	QType MQ[C2B][FILTER_SIZE][FILTER_SIZE][C2B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C2B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}



	QType tmp = 0;

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL1 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL1 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}

void ReLU_layer2(QType P[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_PL1; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_PL1; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void maxpool_layer2(QType N[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B], QType P[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C2B]) {


	QType tmp = 0;

    for (int o = 0; o < C2B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_PL1 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL1 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_ALL; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel
    }
}



// For layer 3
void conv2D_layer3_first(QType N[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C2B],
		QType P[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B], const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B], DataType scale_x[1], DataType zp_x[1]) {

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
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
	QType MQ[C3B][FILTER_SIZE][FILTER_SIZE][C2B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C2B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}


	QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL2  - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}


void conv2D_layer3(QType N[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B],
		QType P[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B], const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C3B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_PL2_PAD][OUT_IMAGE_SIZE_PL2_PAD][C3B];
	for (int i = 0; i < C3B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_PL2; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_PL2; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}
	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
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
	QType MQ[C3B][FILTER_SIZE][FILTER_SIZE][C3B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}



	QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL2 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL2 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}

void conv2D_layer3_break(QType N[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B],
		QType P[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B], const DataType M[C3B * FILTER_SIZE * FILTER_SIZE * C3B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_PL2_PAD][OUT_IMAGE_SIZE_PL2_PAD][C3B];
	for (int i = 0; i < C3B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_PL2; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_PL2; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (min_val > M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
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
	QType MQ[C3B][FILTER_SIZE][FILTER_SIZE][C3B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C3B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}


	QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL2 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL2 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                            tmp += tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * (QType)MQ[o*FILTER_SIZE*FILTER_SIZE*C3B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}




void ReLU_layer3(QType P[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_PL2; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_PL2; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void maxpool_layer3(QType N[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B], QType P[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C3B]) {


	QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_PL2 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL2 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_ALL; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel
    }
}



// For layer 4
void conv2D_layer4_first(QType N[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C3B],
		QType P[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B], const DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C3B], DataType scale_x[1], DataType zp_x[1]) {


	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (min_val > M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
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
	QType MQ[C4B][FILTER_SIZE][FILTER_SIZE][C3B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}
	QType tmp = 0;

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL3  - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL3  - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(N[i+m][j+n][l]  - zp_x[0]) * (QType)MQ[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}


void conv2D_layer4(QType N[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B],
		QType P[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B], const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C4B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_PL3+pad][OUT_IMAGE_SIZE_PL3+pad][C4B];
	for (int i = 0; i < C4B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_PL3; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_PL3; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					if (min_val > M[i][j][k][l])
						min_val = M[i][j][k][l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
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
	QType MQ[C4B][FILTER_SIZE][FILTER_SIZE][C3B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C3B; l++){
					mqpoint = zp_w + M[i][j][k][l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}


	QType tmp = 0;

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL3 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL3 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}

void conv2D_layer4_break(QType N[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B],
		QType P[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B], const DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C4B], DataType scale_x[1], DataType zp_x[1]) {

	// For this we need padding
	int pad = 2;
	QType local_padded[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C4B]; // reusing for PAD
	for (int i = 0; i < C4B; i++){ // image channels

	        for (int j = 0; j < OUT_IMAGE_SIZE_PL3; j++){              // rows

	            for (int k = 0; k < OUT_IMAGE_SIZE_PL3; k++){          // columns
	            	local_padded[j][k][i] = N[j][k][i];
	            }
	        }
	}
	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C4B; l++){
					if (min_val > M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C4B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C4B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l];
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
	QType MQ[C4B][FILTER_SIZE][FILTER_SIZE][C4B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[3], stat[2]);
	DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

	for(int i = 0; i < C4B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C4B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C3B + j*FILTER_SIZE*FILTER_SIZE + k*FILTER_SIZE + l] / scale_w;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_x[0] * scale_w/scale_next)*(mqpoint - zp_w);
					MQ[i][j][k][l] = QType(mqpoint);
				}
			}
		}
	}

	QType tmp = 0;

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_PL3 + pad - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL3 + pad - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)(local_padded[i+m][j+n][l] - zp_x[0]) * (QType)MQ[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
                        }
                    }
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + zero_point_next;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
                scale_x[0] = scale_next;
                zp_x[0] = zero_point_next;
}


void ReLU_layer4(QType P[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_PL3; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_PL3; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void maxpool_layer4(QType N[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B], QType P[OUT_IMAGE_SIZE_PL4][OUT_IMAGE_SIZE_PL4][C4B]) {


	QType tmp = 0;

    for (int o = 0; o < C4B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_PL3 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_PL3 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_ALL; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel
    }
}



// End for all the layers


void fc1(DataType N[F1B],
		DataType P[FOUT]) {

// Create local filter
const DataType M[FOUT][F1B] = {
		#include "ff.txt"
		};
DataType tmp = 0.0;

    for (int i = 0; i < FOUT; i++){     // image rows
                        
    	tmp = 0.0;
		for (int j = 0; j < F1B; j++){ //  image columns
			// actual multiply and add
			tmp += N[j] * M[i][j];
		}
		P[i] = tmp;
		tmp = 0.0;
	}
}



// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType Ps[FOUT],
		DataType ML31C1[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML31C2[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C1[C4B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C1[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B]) {

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp
//#pragma HLS INTERFACE mode=axis port=N_c1
////#pragma HLS INTERFACE mode=axis port=Ps
//	// Local buffer for input image

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

	QType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B];
	conv2D_c1(INP, local_relu_1, scale_x, zp_x);
	ReLU1(local_relu_1);
	QType local_p2[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
	maxpool_1(local_relu_1, local_p2);

	// Layer 1

	const DataType ML10C1Q[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
			                                                                         #include "c2f.txt"
			     };

	QType local_l10c1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
    conv2D_layer1(local_p2,local_l10c1, ML10C1Q, scale_x, zp_x);
    ReLU_layer1(local_l10c1);
    const DataType ML10C2[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
    			                                                                         #include "c3f.txt"
    			     };

	QType local_l10c2[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
    conv2D_layer1(local_l10c1,local_l10c2, ML10C2, scale_x, zp_x);

	const DataType ML11C1[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
			                                                                         #include "c4f.txt"
			     };
	QType local_l11c1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
    conv2D_layer1(local_l10c2,local_l11c1, ML11C1, scale_x, zp_x);
    ReLU_layer1(local_l11c1);
    const DataType ML11C2[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
    			                                                                         #include "c5f.txt"
    			     };
    QType local_l11c2[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
    conv2D_layer1(local_l11c1,local_l11c2, ML11C2, scale_x, zp_x);
    ReLU_layer1(local_l11c2);

    // Layer 1 ends

    // Intermediate downsample
    QType local_layer1_pool[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C1B];
    maxpool_layer1(local_l11c2, local_layer1_pool);


    // Layer 2

	const DataType ML20C1[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
																					 #include "c6f.txt"
				 };
	QType local_l20c1[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B];
	conv2D_layer2_first(local_layer1_pool,local_l20c1, ML20C1, scale_x, zp_x);
	ReLU_layer2(local_l20c1);
	const DataType ML20C2[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
																						 #include "c7f.txt"
					 };
	QType local_l20c2[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B];
	conv2D_layer2(local_l20c1,local_l20c2, ML20C2, scale_x, zp_x);

	const DataType ML21C1[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
																					 #include "c8f.txt"
				 };
	QType local_l21c1[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B];
	conv2D_layer2(local_l20c2,local_l21c1, ML21C1, scale_x, zp_x);
	ReLU_layer2(local_l21c1);
	const DataType ML21C2[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
																						 #include "c9f.txt"
					 };
	QType local_l21c2[OUT_IMAGE_SIZE_PL1][OUT_IMAGE_SIZE_PL1][C2B];
	conv2D_layer2(local_l21c1,local_l21c2, ML21C2, scale_x, zp_x);
	ReLU_layer2(local_l21c2);

	// Layer 2 ends


    // Intermediate downsample
	QType local_layer2_pool[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C2B];
    maxpool_layer2(local_l21c2, local_layer2_pool);


    // Layer 3

	const DataType ML30C1[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
																					 #include "c10f.txt"
				 };
	QType local_l30c1[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B];
	conv2D_layer3_first(local_layer2_pool,local_l30c1, ML30C1, scale_x, zp_x);
	ReLU_layer3(local_l30c1);
	const DataType ML30C2[C3B][FILTER_SIZE][FILTER_SIZE][C3B] = {
																						 #include "c11f.txt"
					 };
	QType local_l30c2[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B];
	conv2D_layer3(local_l30c1,local_l30c2, ML30C2, scale_x, zp_x);

	// Till this point memory problem does not happen


	QType local_l31c1[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B];
	conv2D_layer3_break(local_l30c2,local_l31c1, ML31C1, scale_x, zp_x);
	ReLU_layer3(local_l31c1);
	QType local_l31c2[OUT_IMAGE_SIZE_PL2][OUT_IMAGE_SIZE_PL2][C3B];
	conv2D_layer3_break(local_l31c1,local_l31c2, ML31C2, scale_x, zp_x);
	ReLU_layer3(local_l31c2);

	// Layer 3 ends

    // Intermediate downsample
	QType local_layer3_pool[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C3B];
    maxpool_layer3(local_l31c2, local_layer3_pool);


    // Layer 4


    QType local_l40c1[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B];
	conv2D_layer4_first(local_layer3_pool,local_l40c1, ML40C1, scale_x, zp_x);
	ReLU_layer4(local_l40c1);
	QType local_l40c2[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B];
	conv2D_layer4_break(local_l40c1,local_l40c2, ML40C2, scale_x, zp_x);

	QType local_l41c1[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B];
	conv2D_layer4_break(local_l40c2,local_l41c1, ML41C1, scale_x, zp_x);
	ReLU_layer4(local_l41c1);
	QType local_l41c2[OUT_IMAGE_SIZE_PL3][OUT_IMAGE_SIZE_PL3][C4B];
	conv2D_layer4_break(local_l41c1,local_l41c2, ML41C2, scale_x, zp_x);
	ReLU_layer4(local_l41c2);

	// Layer 4 ends


	// Intermediate downsample
	QType local_layer4_pool[OUT_IMAGE_SIZE_PL4][OUT_IMAGE_SIZE_PL4][C4B];
	maxpool_layer4(local_l41c2, local_layer4_pool);

	DataType local_fc[F1B];

    // Flatten buffer
    for (int k = 0; k < C4B; k++){     // image channels
        // actual multiply and add
    	local_fc[k] = local_layer4_pool[0][0][k];
    }

    // Fully connected ones
    DataType local_fc_out[FOUT];

    fc1(local_fc, local_fc_out);

    // finally copy this this to output
    for (int k = 0; k < FOUT; k++){
        Ps[k] = local_fc_out[k];
        //cout << "Output:" << Ps[k] << endl;
    }
}

