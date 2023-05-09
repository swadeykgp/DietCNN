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


typedef float DataType;

typedef ap_uint<8> QType;

using namespace std;

static const DataType stat[] = { 38.0929, -34.3413, 104.3430, 0, 107.0603, 0, 74.4338, 0, 52.3959, 0, 24.7309, 0, 31.2128, 0, 62.8690, 0};

int compare_integers (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

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


void conv2D_c1(QType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };

/////////////////////////////// START QUATIZATION ///////////////////////////////

// Modify and quantize the bias
// Find min value
DataType min_val;
for(int i = 0; i < C1B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
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
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
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
QType MQ[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL];
DataType mqpoint;
// Process X the input - remember this is CONV 1 - use stats of CONV 2
DataType scale_next = calc_scale(stat[3], stat[2]);
DataType zero_point_next = calc_zero_point(stat[3], stat[2], scale_next);

for(int i = 0; i < C1B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
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

const DataType B[C1B] = {
		#include "c1b.txt"
		};
for(int i = 0; i < C1B; i++){
				if (min_val > B[i])
					min_val = B[i];
}
// Find max value
for(int i = 0; i < C1B; i++){
				if (max_val < B[i])
					max_val = B[i];
}

qmin = 0;
qmax = pow(2, nb) - 1;

DataType scale_b, zp_b;
scale_b = calc_scale(min_val, max_val);
zp_b = calc_zero_point(min_val, max_val, scale_b);
QType BQ[C1B];

for(int i = 0; i < C1B; i++){
				mqpoint = zp_b + B[i] / scale_b;
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
				BQ[i] = QType(mqpoint);
}





/////////////////////////////// END QUATIZATION ///////////////////////////////

QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - C1B; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - C1B; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                                                
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



void conv2D_c2(QType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B],
		QType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c2f.txt"
		     };

/////////////////////////////// START QUATIZATION ///////////////////////////////

// Modify and quantize the bias
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
DataType scale_next = calc_scale(stat[5], stat[4]);
DataType zero_point_next = calc_zero_point(stat[5], stat[4], scale_next);

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

// Modify and quantize the biases
// Find min value
// Create local bias

const DataType B[C2B] = {
		#include "c2b.txt"
		};
for(int i = 0; i < C2B; i++){
				if (min_val > B[i])
					min_val = B[i];
}
// Find max value
for(int i = 0; i < C2B; i++){
				if (max_val < B[i])
					max_val = B[i];
}

qmin = 0;
qmax = pow(2, nb) - 1;

DataType scale_b, zp_b;
scale_b = calc_scale(min_val, max_val);
zp_b = calc_zero_point(min_val, max_val, scale_b);
QType BQ[C2B];

for(int i = 0; i < C2B; i++){
				mqpoint = zp_b + B[i] / scale_b;
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
				BQ[i] = QType(mqpoint);
}





/////////////////////////////// END QUATIZATION ///////////////////////////////

QType tmp = 0;
    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                            tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c3(QType N[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C2B],
		QType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c3f.txt"
		     };
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
DataType scale_next = calc_scale(stat[7], stat[6]);
DataType zero_point_next = calc_zero_point(stat[7], stat[6], scale_next);

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

// Modify and quantize the biases
// Find min value
// Create local bias

const DataType B[C3B] = {
		#include "c3b.txt"
		};
for(int i = 0; i < C3B; i++){
				if (min_val > B[i])
					min_val = B[i];
}
// Find max value
for(int i = 0; i < C3B; i++){
				if (max_val < B[i])
					max_val = B[i];
}

qmin = 0;
qmax = pow(2, nb) - 1;

DataType scale_b, zp_b;
scale_b = calc_scale(min_val, max_val);
zp_b = calc_zero_point(min_val, max_val, scale_b);
QType BQ[C3B];

for(int i = 0; i < C3B; i++){
				mqpoint = zp_b + B[i] / scale_b;
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
				BQ[i] = QType(mqpoint);
}





/////////////////////////////// END QUATIZATION ///////////////////////////////

QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                            tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c4(QType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C3B],
		QType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C3B] = {
		                                                                         #include "c4f.txt"
		     };
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
DataType scale_next = calc_scale(stat[9], stat[8]);
DataType zero_point_next = calc_zero_point(stat[9], stat[8], scale_next);

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

// Modify and quantize the biases
// Find min value
// Create local bias

const DataType B[C4B] = {
		#include "c4b.txt"
		};
for(int i = 0; i < C4B; i++){
				if (min_val > B[i])
					min_val = B[i];
}
// Find max value
for(int i = 0; i < C4B; i++){
				if (max_val < B[i])
					max_val = B[i];
}

qmin = 0;
qmax = pow(2, nb) - 1;

DataType scale_b, zp_b;
scale_b = calc_scale(min_val, max_val);
zp_b = calc_zero_point(min_val, max_val, scale_b);
QType BQ[C4B];

for(int i = 0; i < C4B; i++){
				mqpoint = zp_b + B[i] / scale_b;
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
				BQ[i] = QType(mqpoint);
}





/////////////////////////////// END QUATIZATION ///////////////////////////////

QType tmp = 0;


    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                            tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c5(QType N[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C4B],
		QType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C5B][FILTER_SIZE][FILTER_SIZE][C4B] = {
		                                                                         #include "c5f.txt"
		     };
//////////////////////////////// START QUATIZATION ///////////////////////////////

// Modify and quantize the bias
// Find min value
DataType min_val;
for(int i = 0; i < C5B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C4B; l++){
				if (min_val > M[i][j][k][l])
					min_val = M[i][j][k][l];
			}
		}
	}
}
// Find max value
DataType max_val;
for(int i = 0; i < C5B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C4B; l++){
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
QType MQ[C5B][FILTER_SIZE][FILTER_SIZE][C4B];
DataType mqpoint;
// Process X the input - remember this is CONV 1 - use stats of CONV 2
DataType scale_next = calc_scale(stat[11], stat[10]);
DataType zero_point_next = calc_zero_point(stat[11], stat[10], scale_next);

for(int i = 0; i < C5B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C4B; l++){
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

const DataType B[C5B] = {
		#include "c5b.txt"
		};
for(int i = 0; i < C2B; i++){
				if (min_val > B[i])
					min_val = B[i];
}
// Find max value
for(int i = 0; i < C5B; i++){
				if (max_val < B[i])
					max_val = B[i];
}

qmin = 0;
qmax = pow(2, nb) - 1;

DataType scale_b, zp_b;
scale_b = calc_scale(min_val, max_val);
zp_b = calc_zero_point(min_val, max_val, scale_b);
QType BQ[C5B];

for(int i = 0; i < C5B; i++){
				mqpoint = zp_b + B[i] / scale_b;
				mqpoint = mqpoint < qmin ? qmin : mqpoint;
				mqpoint = mqpoint > qmax ? qmax : mqpoint;
				mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
				BQ[i] = QType(mqpoint);
}





/////////////////////////////// END QUATIZATION ///////////////////////////////

QType tmp = 0;

    for (int o = 0; o < C5B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                            tmp += QType((N[i+m][j+n][l] - zp_x[0]) * M[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c6(QType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C5B],
		QType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], DataType M[C6B * FILTER_SIZE * FILTER_SIZE * C5B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C6B][FILTER_SIZE][FILTER_SIZE][C5B] = {
//		                                                                         #include "c6f.txt"
//		     };



	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C6B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C5B; l++){
					if (min_val > M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C6B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C5B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
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
	QType MQ[C6B][FILTER_SIZE][FILTER_SIZE][C5B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[13], stat[12]);
	DataType zero_point_next = calc_zero_point(stat[13], stat[12], scale_next);

	for(int i = 0; i < C6B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C5B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] / scale_w;
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

	const DataType B[C6B] = {
			#include "c6b.txt"
			};
	for(int i = 0; i < C6B; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < C6B; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[C6B];

	for(int i = 0; i < C6B; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}





	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int o = 0; o < C6B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C5B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	// actual multiply and add replaced by table lookup
                            //tmp += N[i+m][j+n][l] * M[o][m][n][l];
                            tmp +=  QType((N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c7(QType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C6B],
		QType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B], DataType M[C7B * FILTER_SIZE * FILTER_SIZE * C6B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C7B][FILTER_SIZE][FILTER_SIZE][C6B] = {
//		                                                                         #include "c7f.txt"
//		     };

	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C7B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C6B; l++){
					if (min_val >  M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C7B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C6B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
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
	QType MQ[C7B][FILTER_SIZE][FILTER_SIZE][C6B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[15], stat[14]);
	DataType zero_point_next = calc_zero_point(stat[15], stat[14], scale_next);

	for(int i = 0; i < C7B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C6B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] / scale_w;
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

	const DataType B[C7B] = {
			#include "c7b.txt"
			};
	for(int i = 0; i < C7B; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < C7B; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[C7B];

	for(int i = 0; i < C7B; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}





	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int o = 0; o < C7B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C6B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	//tmp += N[i+m][j+n][l] * M[o][m][n][l];
                        	tmp +=  QType((N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
    // fill the scale variables
        scale_x[0] = scale_next;
        zp_x[0] = zero_point_next;
}

void conv2D_c8(QType N[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C7B],
		QType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B], DataType M[C8B * FILTER_SIZE * FILTER_SIZE * C7B], DataType scale_x[1], DataType zp_x[1]) {

// Create local filter
// This one is full implementation with channel



	/////////////////////////////// START QUATIZATION ///////////////////////////////

	// Modify and quantize the bias
	// Find min value
	DataType min_val;
	for(int i = 0; i < C8B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C7B; l++){
					if (min_val > M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						min_val = M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
				}
			}
		}
	}
	// Find max value
	DataType max_val;
	for(int i = 0; i < C8B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C7B; l++){
					if (max_val < M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k])
						max_val = M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k];
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
	QType MQ[C8B][FILTER_SIZE][FILTER_SIZE][C7B];
	DataType mqpoint;
	// Process X the input - remember this is CONV 1 - use stats of CONV 2
	DataType scale_next = calc_scale(stat[15], stat[14]);
	DataType zero_point_next = calc_zero_point(stat[15], stat[14], scale_next);

	for(int i = 0; i < C8B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C7B; l++){
					mqpoint = zp_w + M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k] / scale_w;
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

	const DataType B[C8B] = {
			#include "c8b.txt"
			};
	for(int i = 0; i < C8B; i++){
					if (min_val > B[i])
						min_val = B[i];
	}
	// Find max value
	for(int i = 0; i < C8B; i++){
					if (max_val < B[i])
						max_val = B[i];
	}

	qmin = 0;
	qmax = pow(2, nb) - 1;

	DataType scale_b, zp_b;
	scale_b = calc_scale(min_val, max_val);
	zp_b = calc_zero_point(min_val, max_val, scale_b);
	QType BQ[C8B];

	for(int i = 0; i < C8B; i++){
					mqpoint = zp_b + B[i] / scale_b;
					mqpoint = mqpoint < qmin ? qmin : mqpoint;
					mqpoint = mqpoint > qmax ? qmax : mqpoint;
					mqpoint = (scale_b/scale_next)*(mqpoint + zp_b);
					BQ[i] = QType(mqpoint);
	}





	/////////////////////////////// END QUATIZATION ///////////////////////////////

	QType tmp = 0;

    for (int o = 0; o < C8B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C7B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	//tmp += N[i+m][j+n][l] * M[o][m][n][l];
                        	tmp +=  QType((N[i+m][j+n][l] - zp_x[0]) * MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
            	//cout << "Putting value" << tmp << endl;
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] = tmp + BQ[o] + zero_point_next;
                tmp = 0.0;
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
void ReLU2(QType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}
void ReLU3(QType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C3; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C3; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}
void ReLU4(QType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C4; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C4; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}
void ReLU5(QType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B]) {
	int ii = 0;
	for (int k = 0; k < C5B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C5; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C5; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}
void ReLU6(QType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B]) {
	int ii = 0;
	for (int k = 0; k < C6B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C6; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C6; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void ReLU7(QType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B]) {
	int ii = 0;
	for (int k = 0; k < C7B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C7; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C7; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}

void ReLU8(QType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B]) {
	int ii = 0;
	for (int k = 0; k < C8B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C8; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C8; j++){          // columns

				P[i][j][k] = ((QType)0 < (QType)P[i][j][k]) ? (QType)P[i][j][k] : (QType)0;
			}
		}
	}
}


void fc1(DataType N[OUT_IMAGE_SIZE_F1_IN],
		DataType P[OUT_IMAGE_SIZE_F1_OUT]) {

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN] = {
		#include "f1f.txt"
		};
const DataType B[OUT_IMAGE_SIZE_F1_OUT] = {
		#include "f1b.txt"
		};
DataType tmp = 0.0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
                        
    	tmp = 0.0;
		for (int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){ //  image columns
			// actual multiply and add
			tmp += N[j] * M[i][j];
		}
		P[i] = tmp;
		tmp = 0.0;
	}
}

void maxpool_1(QType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B]) {


DataType tmp = 0.0;

    for (int o = 0; o < C1B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_C1; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0.0;
            }
        } // end of one output channel
    }
}

void maxpool_2(QType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B], QType P[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][C2B]) {


DataType tmp = 0.0;

    for (int o = 0; o < C2B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_C1; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0.0;
            }
        } // end of one output channel
    }
}


void maxpool_3(QType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B], QType P[OUT_IMAGE_SIZE_P3][OUT_IMAGE_SIZE_P3][C4B]) {


DataType tmp = 0.0;

    for (int o = 0; o < C4B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_C1; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0.0;
            }
        } // end of one output channel
    }
}

void maxpool_4(QType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], QType P[OUT_IMAGE_SIZE_P4][OUT_IMAGE_SIZE_P4][C6B]) {


DataType tmp = 0.0;

    for (int o = 0; o < C6B; o++){ // image channels

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - STRIDE_C1; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - STRIDE_C1; j=j+STRIDE_C1){          // columns

                for (int m = 0; m < STRIDE_C1; m++){     // kernel rows

                        for (int n = 0; n < STRIDE_C1; n++){ // kernel columns
                            // max pooling operation
                        	if (N[i+m][j+n][o] > tmp)
                        	    tmp = N[i+m][j+n][o];
                        }
                } // end of one maxpool window
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0.0;
            }
        } // end of one output channel
    }
}

void maxpool_5(QType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B], QType P[OUT_IMAGE_SIZE_P5][OUT_IMAGE_SIZE_P5][C8B]) {


DataType tmp = 0.0;

    for (int o = 0; o < C8B; o++){ // image channels
    	tmp = 0.0;
    	if(N[0][0][o] > tmp)
    		tmp = N[0][0][o];
    	else if(N[0][1][o] > tmp)
    		tmp = N[0][1][o];
    	else if(N[1][0][o] > tmp)
    	    tmp = N[1][0][o];
    	else
    	    tmp = N[1][1][o];
		P[0][0][o] =  tmp;
		tmp = 0.0;
    }
}

// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
	DataType Ps[OUT_IMAGE_SIZE_F1_OUT],
	DataType M1[C6B * FILTER_SIZE * FILTER_SIZE * C5B],
	DataType M2[C7B * FILTER_SIZE * FILTER_SIZE * C6B],
	DataType M3[C8B * FILTER_SIZE * FILTER_SIZE * C7B]) {

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
	QType local_pool_1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
	maxpool_1(local_relu_1, local_pool_1);

	QType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B];
    conv2D_c2(local_pool_1, local_conv_2, scale_x, zp_x);
    ReLU2(local_conv_2);
    QType local_pool_2[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][C2B];
    maxpool_2(local_conv_2, local_pool_2);

    QType local_conv_3[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B];
    conv2D_c3(local_pool_2, local_conv_3, scale_x, zp_x);
    ReLU3(local_conv_3);

    QType local_conv_4[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B];
    conv2D_c4(local_conv_3, local_conv_4, scale_x, zp_x);
    ReLU4(local_conv_4);
    QType local_pool_3[OUT_IMAGE_SIZE_P3][OUT_IMAGE_SIZE_P3][C3B];
    maxpool_3(local_conv_4, local_pool_3);

    QType local_conv_5[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B];
    conv2D_c5(local_pool_3, local_conv_5, scale_x, zp_x);
    ReLU5(local_conv_5);

    QType local_conv_6[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B];
    conv2D_c6(local_conv_5, local_conv_6, M1, scale_x, zp_x);
    ReLU6(local_conv_6);
    QType local_pool_4[OUT_IMAGE_SIZE_P4][OUT_IMAGE_SIZE_P4][C6B];
    maxpool_4(local_conv_6, local_pool_4);

    QType local_conv_7[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B];
    conv2D_c7(local_pool_4, local_conv_7, M2, scale_x, zp_x);
    ReLU7(local_conv_7);

    QType local_conv_8[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B];
    conv2D_c8(local_conv_7, local_conv_8, M3, scale_x, zp_x);
    ReLU8(local_conv_8);

    QType local_pool_5[OUT_IMAGE_SIZE_P5][OUT_IMAGE_SIZE_P5][C8B];
    maxpool_5(local_conv_7, local_pool_5);
    DataType local_fc_1[OUT_IMAGE_SIZE_F1_IN];


    // Flatten buffer
    for (int k = 0; k < C8B; k++){     // image channels
        // actual multiply and add
        local_fc_1[k] = local_pool_5[0][0][k];
    }

    // Fully connected ones
    DataType local_fc_2[OUT_IMAGE_SIZE_F1_OUT];

    fc1(local_fc_1, local_fc_2);

    // finally copy this this to output
    for (int k = 0; k < OUT_IMAGE_SIZE_F1_OUT; k++){
        Ps[k] = local_fc_2[k];
    }
}
