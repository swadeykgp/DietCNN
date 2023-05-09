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
#include <ap_fixed.h>


typedef ap_fixed<16,6> QType;


typedef float DataType;

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
		QType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };


QType MQ[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL];

for(int i = 0; i < C1B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < INP_IMAGE_CHANNEL; l++){
				MQ[i][j][k][l] = QType(M[i][j][k][l]);
			}
		}
	}
}

const DataType B[C1B] = {
		#include "c1b.txt"
		};

QType BQ[C1B];

for(int i = 0; i < C1B; i++){
				BQ[i] = QType(B[i]);
}


QType tmp = 0;

    for (int o = 0; o < C1B; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - C1B; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - C1B; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                                                
                            // actual multiply and add
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);
                            tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}



void conv2D_c2(QType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B],
		QType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c2f.txt"
		     };


QType MQ[C2B][FILTER_SIZE][FILTER_SIZE][C1B];

for(int i = 0; i < C2B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C1B; l++){
				MQ[i][j][k][l] = QType(M[i][j][k][l]);
			}
		}
	}
}


const DataType B[C2B] = {
		#include "c2b.txt"
		};
QType BQ[C2B];

for(int i = 0; i < C2B; i++){
				BQ[i] = QType(B[i]);
}
QType tmp = 0;
    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);
                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c3(QType N[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C2B],
		QType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c3f.txt"
		     };

QType MQ[C3B][FILTER_SIZE][FILTER_SIZE][C2B];
for(int i = 0; i < C3B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C2B; l++){
				MQ[i][j][k][l] = QType(M[i][j][k][l]);
			}
		}
	}
}

const DataType B[C3B] = {
		#include "c3b.txt"
		};
QType BQ[C3B];

for(int i = 0; i < C3B; i++){
				BQ[i] = QType(B[i]);
}

QType tmp = 0;

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);
                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c4(QType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C3B],
		QType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C3B] = {
		                                                                         #include "c4f.txt"
		     };
QType MQ[C4B][FILTER_SIZE][FILTER_SIZE][C3B];

for(int i = 0; i < C4B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C3B; l++){
				MQ[i][j][k][l] = QType(M[i][j][k][l]);
			}
		}
	}
}

const DataType B[C4B] = {
		#include "c4b.txt"
		};

QType BQ[C4B];

for(int i = 0; i < C4B; i++){
				BQ[i] = QType(B[i]);
}

QType tmp = 0;

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);
                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c5(QType N[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C4B],
		QType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C5B][FILTER_SIZE][FILTER_SIZE][C4B] = {
		                                                                         #include "c5f.txt"
		     };

QType MQ[C5B][FILTER_SIZE][FILTER_SIZE][C4B];
for(int i = 0; i < C5B; i++){
	for(int j = 0; j < FILTER_SIZE; j++){
		for(int k = 0; k < FILTER_SIZE; k++){
			for(int l = 0; l < C4B; l++){
				MQ[i][j][k][l] = QType(M[i][j][k][l]);
			}
		}
	}
}
const DataType B[C5B] = {
		#include "c5b.txt"
		};
QType BQ[C5B];

for(int i = 0; i < C5B; i++){
				BQ[i] = QType(B[i]);
}

QType tmp = 0;

    for (int o = 0; o < C5B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);
                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c6(QType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C5B],
		QType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], DataType M[C6B * FILTER_SIZE * FILTER_SIZE * C5B]) {


	QType MQ[C6B][FILTER_SIZE][FILTER_SIZE][C5B];
	for(int i = 0; i < C6B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C5B; l++){
					MQ[i][j][k][l] = QType(M[i*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k]);
				}
			}
		}
	}

	const DataType B[C6B] = {
			#include "c6b.txt"
			};

	QType BQ[C6B];

	for(int i = 0; i < C6B; i++){
					BQ[i] = QType(B[i]);
	}

	QType tmp = 0;

    for (int o = 0; o < C6B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C5B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);

                            tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c7(QType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C6B],
		QType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B], DataType M[C7B * FILTER_SIZE * FILTER_SIZE * C6B]) {

	QType MQ[C7B][FILTER_SIZE][FILTER_SIZE][C6B];
	for(int i = 0; i < C7B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C6B; l++){
					MQ[i][j][k][l] = QType(M[i*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k]);
				}
			}
		}
	}

	const DataType B[C7B] = {
			#include "c7b.txt"
			};
	QType BQ[C7B];

	for(int i = 0; i < C7B; i++){
					BQ[i] = QType(B[i]);
	}
	QType tmp = 0;

    for (int o = 0; o < C7B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C6B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);

                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c8(QType N[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C7B],
		QType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B], DataType M[C8B * FILTER_SIZE * FILTER_SIZE * C7B]) {

	QType MQ[C8B][FILTER_SIZE][FILTER_SIZE][C7B];

	for(int i = 0; i < C8B; i++){
		for(int j = 0; j < FILTER_SIZE; j++){
			for(int k = 0; k < FILTER_SIZE; k++){
				for(int l = 0; l < C7B; l++){
					MQ[i][j][k][l] = QType(M[i*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + j*FILTER_SIZE + k]);
				}
			}
		}
	}

	const DataType B[C8B] = {
			#include "c8b.txt"
			};
	QType BQ[C8B];

	for(int i = 0; i < C8B; i++){
					BQ[i] = QType(B[i]);
	}

	QType tmp = 0;

    for (int o = 0; o < C8B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C7B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                        	tmp += (QType)( N[i+m][j+n][l] >> MQ[o][m][n][l]);

                        	tmp += (QType)( N[i+m][j+n][l] - MQ[o][m][n][l]);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
            	//cout << "Putting value" << tmp << endl;
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] = tmp + BQ[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
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
			//tmp += N[j] >> M[i][j];
			tmp += N[j] - M[i][j];
		}
		P[i] = tmp;
		tmp = 0.0;
	}
}

void maxpool_1(QType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], QType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B]) {


	QType tmp = 0.0;

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


	QType tmp = 0.0;

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


	QType tmp = 0.0;

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


	QType tmp = 0.0;

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


QType tmp = 0.0;

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

	QType INP[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL];
	for(int i = 0; i < INP_IMAGE_SIZE; i++){
		for(int j = 0; j < INP_IMAGE_SIZE; j++){
			for(int k = 0; k < INP_IMAGE_CHANNEL; k++){
				INP[i][j][k] = QType(N_c1[i][j][k]);
			}
		}
	}

	QType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B];
	conv2D_c1(INP, local_relu_1);
	ReLU1(local_relu_1);
	QType local_pool_1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
	maxpool_1(local_relu_1, local_pool_1);

	QType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B];
    conv2D_c2(local_pool_1, local_conv_2);
    ReLU2(local_conv_2);
    QType local_pool_2[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][C2B];
    maxpool_2(local_conv_2, local_pool_2);

    QType local_conv_3[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B];
    conv2D_c3(local_pool_2, local_conv_3);
    ReLU3(local_conv_3);

    QType local_conv_4[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B];
    conv2D_c4(local_conv_3, local_conv_4);
    ReLU4(local_conv_4);
    QType local_pool_3[OUT_IMAGE_SIZE_P3][OUT_IMAGE_SIZE_P3][C3B];
    maxpool_3(local_conv_4, local_pool_3);

    QType local_conv_5[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B];
    conv2D_c5(local_pool_3, local_conv_5);
    ReLU5(local_conv_5);

    QType local_conv_6[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B];
    conv2D_c6(local_conv_5, local_conv_6, M1);
    ReLU6(local_conv_6);
    QType local_pool_4[OUT_IMAGE_SIZE_P4][OUT_IMAGE_SIZE_P4][C6B];
    maxpool_4(local_conv_6, local_pool_4);

    QType local_conv_7[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B];
    conv2D_c7(local_pool_4, local_conv_7, M2);
    ReLU7(local_conv_7);

    QType local_conv_8[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B];
    conv2D_c8(local_conv_7, local_conv_8, M3);
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
