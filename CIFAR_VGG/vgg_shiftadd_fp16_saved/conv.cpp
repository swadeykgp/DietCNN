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


typedef ap_fixed<16,6> DataType;

using namespace std;

int compare_integers (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


void conv2D_c1(const DataType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };

// Create local bias
const DataType B[C1B] = {
		#include "c1b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;

    for (int o = 0; o < C1B; o++){ // output filter
        
        for (int i = 0; i < INP_IMAGE_SIZE - C1B; i=i+STRIDE_C1){              // rows
        
            for (int j = 0; j < INP_IMAGE_SIZE - C1B; j=j+STRIDE_C1){          // columns
                   
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
                                
                    for (int m = 0; m < FILTER_SIZE; m++){     // kernel rows
                                        
                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
                                                
                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);

							// actual L1 Norm , sum of absolute difference between points in filter
							tmp += (DataType)abs((float)(N[i+m][j+n][l] - M[o][m][n][l]));
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}



void conv2D_c2(DataType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B],
		DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c2f.txt"
		     };

// Create local bias
const DataType B[C2B] = {
		#include "c2b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);

				// actual L1 Norm , sum of absolute difference between points in filter
				tmp += (DataType)abs((float)(N[i+m][j+n][l] - M[o][m][n][l]));
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c3(DataType N[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C2B],
		DataType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c3f.txt"
		     };
// Create local bias
const DataType B[C3B] = {
		#include "c3b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);

				// actual L1 Norm , sum of absolute difference between points in filter
				tmp += (DataType)abs((float)(N[i+m][j+n][l] - M[o][m][n][l]));
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c4(DataType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C3B],
		DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C3B] = {
		                                                                         #include "c4f.txt"
		     };
// Create local bias
const DataType B[C4B] = {
		#include "c4b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c5(DataType N[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C4B],
		DataType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C5B][FILTER_SIZE][FILTER_SIZE][C4B] = {
		                                                                         #include "c5f.txt"
		     };
// Create local bias
const DataType B[C5B] = {
		#include "c5b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C5B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o][m][n][l];
							tmp += (tmp_shift1 >> tmp_shift2);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c6(DataType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C5B],
		DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], DataType M[C6B * FILTER_SIZE * FILTER_SIZE * C5B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C6B][FILTER_SIZE][FILTER_SIZE][C5B] = {
//		                                                                         #include "c6f.txt"
//		     };



// Create local bias
const DataType B[C6B] = {
		#include "c6b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C6B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C5B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
							tmp += (tmp_shift1 >> tmp_shift2);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c7(DataType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C6B],
		DataType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B], DataType M[C7B * FILTER_SIZE * FILTER_SIZE * C6B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C7B][FILTER_SIZE][FILTER_SIZE][C6B] = {
//		                                                                         #include "c7f.txt"
//		     };

// Create local bias
const DataType B[C7B] = {
		#include "c7b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C7B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C6B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
							tmp += (tmp_shift1 >> tmp_shift2);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}

void conv2D_c8(DataType N[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C7B],
		DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B], DataType M[C8B * FILTER_SIZE * FILTER_SIZE * C7B]) {

// Create local filter
// This one is full implementation with channel




// Create local bias
const DataType B[C8B] = {
		#include "c8b.txt"
		};
DataType tmp = 0.0;
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;
    for (int o = 0; o < C8B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	for (int l = 0; l < C7B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                        	tmp_shift1 = (DataType)N[i+m][j+n][l];
							tmp_shift2 = (DataType)M[o*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n];
							tmp += (tmp_shift1 >> tmp_shift2);
                        }       
                    }   
                } // end of one window , all input channels . output is written here, per pixel bias added
            	//cout << "Putting value" << tmp << endl;
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp + B[o];
                tmp = 0.0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
}



void ReLU1(DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}
void ReLU2(DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}
void ReLU3(DataType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C3; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C3; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}
void ReLU4(DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C4; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C4; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}
void ReLU5(DataType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B]) {
	int ii = 0;
	for (int k = 0; k < C5B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C5; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C5; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}
void ReLU6(DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B]) {
	int ii = 0;
	for (int k = 0; k < C6B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C6; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C6; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}

void ReLU7(DataType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B]) {
	int ii = 0;
	for (int k = 0; k < C7B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C7; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C7; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
			}
		}
	}
}

void ReLU8(DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B]) {
	int ii = 0;
	for (int k = 0; k < C8B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C8; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C8; j++){          // columns

				P[i][j][k] = ((DataType)0 < (DataType)P[i][j][k]) ? (DataType)P[i][j][k] : (DataType)0;
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
DataType tmp_shift1 = 0.0;
DataType tmp_shift2 = 0.0;

    for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){     // image rows
                        
    	tmp = 0.0;
		for (int j = 0; j < OUT_IMAGE_SIZE_F1_IN; j++){ //  image columns
			tmp_shift1 = (DataType)N[j];
			tmp_shift2 = (DataType)M[i][j];
			tmp += (tmp_shift1 >> tmp_shift2);
			// actual L1 Norm , sum of absolute difference between points in filter
			tmp += (DataType)abs((float)(N[j] - M[i][j]));
		}
		P[i] = tmp;
		tmp = 0.0;
	}
}

void maxpool_1(DataType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], DataType P[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B]) {


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

void maxpool_2(DataType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B], DataType P[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][C2B]) {


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


void maxpool_3(DataType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B], DataType P[OUT_IMAGE_SIZE_P3][OUT_IMAGE_SIZE_P3][C4B]) {


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

void maxpool_4(DataType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], DataType P[OUT_IMAGE_SIZE_P4][OUT_IMAGE_SIZE_P4][C6B]) {


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

void maxpool_5(DataType N[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B], DataType P[OUT_IMAGE_SIZE_P5][OUT_IMAGE_SIZE_P5][C8B]) {


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

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp
#pragma HLS INTERFACE mode=axis port=N_c1
//#pragma HLS INTERFACE mode=axis port=Ps
	// Local buffer for input image

	DataType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B];
	conv2D_c1(N_c1, local_relu_1);
	//cout << "local_relu_1[0][0][0] = " << local_relu_1[0][0][0] << endl;
	ReLU1(local_relu_1);
	DataType local_pool_1[OUT_IMAGE_SIZE_P1][OUT_IMAGE_SIZE_P1][C1B];
	maxpool_1(local_relu_1, local_pool_1);

	DataType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B];
    conv2D_c2(local_pool_1, local_conv_2);
    //cout << "local_conv_2[0][0][0] = " << local_conv_2[0][0][0] << endl;
    ReLU2(local_conv_2);
    DataType local_pool_2[OUT_IMAGE_SIZE_P2][OUT_IMAGE_SIZE_P2][C2B];
    maxpool_2(local_conv_2, local_pool_2);

    DataType local_conv_3[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B];
    conv2D_c3(local_pool_2, local_conv_3);
    ReLU3(local_conv_3);
    //cout << "local_conv_3[0][0][0] = " << local_conv_3[0][0][0] << endl;

    DataType local_conv_4[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B];
    conv2D_c4(local_conv_3, local_conv_4);
    ReLU4(local_conv_4);
    //cout << "local_conv_4[0][0][0] = " << local_conv_4[0][0][0] << endl;
    DataType local_pool_3[OUT_IMAGE_SIZE_P3][OUT_IMAGE_SIZE_P3][C3B];
    maxpool_3(local_conv_4, local_pool_3);

    DataType local_conv_5[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B];
    conv2D_c5(local_pool_3, local_conv_5);
    ReLU5(local_conv_5);
    //cout << "local_conv_5[0][0][0] = " << local_conv_5[0][0][0] << endl;

    DataType local_conv_6[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B];
    conv2D_c6(local_conv_5, local_conv_6, M1);
    ReLU6(local_conv_6);
    //cout << "local_conv_6[0][0][0] = " << local_conv_6[0][0][0] << endl;
    DataType local_pool_4[OUT_IMAGE_SIZE_P4][OUT_IMAGE_SIZE_P4][C6B];
    maxpool_4(local_conv_6, local_pool_4);

    DataType local_conv_7[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B];
    conv2D_c7(local_pool_4, local_conv_7, M2);
    ReLU7(local_conv_7);
    //cout << "local_conv_7[0][0][0] = " << local_conv_7[0][0][0] << endl;

    DataType local_conv_8[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B];
    conv2D_c8(local_conv_7, local_conv_8, M3);
    ReLU8(local_conv_8);
    //cout << "local_conv_8[0][0][0] = " << local_conv_8[0][0][0] << endl;

    DataType local_pool_5[OUT_IMAGE_SIZE_P5][OUT_IMAGE_SIZE_P5][C8B];
    //maxpool_5(local_conv_8, local_pool_5);
    maxpool_5(local_conv_7, local_pool_5);
    //for (int xx=0;xx<200;xx++){
    //    cout << "local_pool_5 = " << local_pool_5[0][0][xx] << endl;
    //}
    //cout << "local_pool_5" << endl;
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
        //cout << "Output:" << Ps[k] << endl;
    }
}

