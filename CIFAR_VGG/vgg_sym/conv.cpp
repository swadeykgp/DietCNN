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
//#include "ap_int.h"

//typedef ap_uint<10> DataType;
//#include <ap_fixed.h>

//typedef ap_fixed<10,0> DataType;
typedef int DataType;
typedef float ReturnType;

using namespace std;

int compare_integers (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


void conv2D_c1(const DataType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1_sym_filter.txt"
		     };
const DataType C1BL[N_CLUSTERS][C1B] = {
							#include "c1b_lut.txt"
		         };


int tmp = 0;
const int tmp_symbol_num = INP_IMAGE_CHANNEL*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

        for (int i = 0; i < INP_IMAGE_SIZE - FILTER_SIZE; i=i+STRIDE_C1){              // rows

            for (int j = 0; j < INP_IMAGE_SIZE - FILTER_SIZE; j=j+STRIDE_C1){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
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
} // conv2d c1


//void conv2D_c1(const DataType N[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
//		DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B],
//		const DataType CL[N_CLUSTERS][N_CFILTERS],
//		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//
//// Create local filter
//// This one is full implementation with channel
//const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][INP_IMAGE_CHANNEL] = {
//		                                                                         #include "c1_sym_filter.txt"
//		     };
//const DataType C1BL[N_CLUSTERS][C1B] = {
//							#include "c1b_lut.txt"
//		         };
//
//
//int tmp = 0;
//const int tmp_symbol_num = INP_IMAGE_CHANNEL*FILTER_SIZE*FILTER_SIZE;
//DataType tmp_conv_sym[tmp_symbol_num];
//
//    for (int o = 0; o < C1B; o++){ // output filter
//
//        for (int i = 0; i < (INP_IMAGE_SIZE/2 +1) - FILTER_SIZE; i=i+STRIDE_ALL){              // rows
//
//            for (int j = 0; j < (INP_IMAGE_SIZE/2 +1) - FILTER_SIZE; j=j+STRIDE_ALL){          // columns
//
//            	tmp = 4096;
//            	DataType tmp_conv_sym[tmp_symbol_num];
//                for (int l = 0; l < INP_IMAGE_CHANNEL; l++){ // image channels
//
//                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows
//
//                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns
//
//                            // actual multiply and add replaced by table lookup
//                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
//                        }
//                    }
//                } // end of one window , all input channels . now ripple addition will start
//
//                // sort temporary symbol array
//                //qsort(tmp_conv_sym, tmp_symbol_num, sizeof(int), compare_integers);
//
//                // Get the first symbol
//                tmp = tmp_conv_sym[0];
//
//                // ripple add next symbol onwards
//                for (int i = 1; i < tmp_symbol_num; i++){
//                	tmp = AL[tmp][tmp_conv_sym[i]];
//                }
//
//                // add bias
//                tmp = C1BL[tmp][o];
//                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
//                tmp = 0;
//            }
//        } // end of one output channel , bias should have been added here. But we add for each out pixel
//    }
//} // conv2d c1

void ReLU1(DataType P[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C1; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C1; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c2(DataType N[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B],
		DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c2_sym_filter.txt"
		     };
const DataType C2BL[N_CLUSTERS][C2B] = {
								#include "c2b_lut.txt"
			     };
int tmp = 0;
const int tmp_symbol_num = C1B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C1 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C1B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU2(DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c3(DataType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B],
		DataType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c3_sym_filter.txt"
		     };
const DataType C3BL[N_CLUSTERS][C3B] = {
									#include "c3b_lut.txt"
				     };
int tmp = 0;
const int tmp_symbol_num = C2B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C2B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
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
                tmp = C3BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 3

void ReLU3(DataType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C3; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C3; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c4(DataType N[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B],
		DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C3B] = {
		                                                                         #include "c4_sym_filter.txt"
		     };
const DataType C4BL[N_CLUSTERS][C4B] = {
									#include "c4b_lut.txt"
				     };
int tmp = 0;
const int tmp_symbol_num = C3B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
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
                tmp = C4BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 4
void ReLU4(DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C4; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C4; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}
void conv2D_c5(DataType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B],
		DataType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C5B][FILTER_SIZE][FILTER_SIZE][C4B] = {
		                                                                         #include "c5_sym_filter.txt"
		     };
const DataType C5BL[N_CLUSTERS][C5B] = {
									#include "c5b_lut.txt"
				     };
int tmp = 0;
const int tmp_symbol_num = C4B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C5B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o][m][n][l]];
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
                tmp = C5BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 5


void ReLU5(DataType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C5B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C5; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C5; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c6(DataType N[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B],
		DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B],
		DataType M[C6B * FILTER_SIZE * FILTER_SIZE * C5B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType C6BL[N_CLUSTERS][C6B] = {
									#include "c6b_lut.txt"
				     };
int tmp = 0;
int nouse = 0;
const int tmp_symbol_num = C5B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

for (int o = 0; o < C6B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
				for (int l = 0; l < C5B ; l++){ // image channels

					for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

						for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

							// actual multiply and add replaced by table lookup
							tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C6B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];

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
                tmp = C6BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 6

void conv2D_c7(DataType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B],
		DataType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B],
		DataType M[C7B * FILTER_SIZE * FILTER_SIZE * C6B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType C7BL[N_CLUSTERS][C7B] = {
									#include "c7b_lut.txt"
				     };
int tmp = 0;
int nouse = 0;
const int tmp_symbol_num = C6B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

for (int o = 0; o < C7B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
            		for (int l = 0; l < C6B ; l++){ // image channels

						for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

							for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

								// actual multiply and add replaced by table lookup
								tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C7B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];

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
                tmp = C7BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 7


void conv2D_c8(DataType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B],
		DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B],
		 DataType M[C8B * FILTER_SIZE * FILTER_SIZE * C7B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType C8BL[N_CLUSTERS][C8B] = {
									#include "c8b_lut.txt"
				     };
int tmp = 0;
const int tmp_symbol_num = C7B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C8B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C7 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C7 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C7B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C8B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                tmp = C8BL[tmp][o];
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                //cout << tmp << "   ";
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 8

void ReLU6(DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C6B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C6; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C6; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void ReLU7(DataType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C7B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C7; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C7; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}





void ReLU8(DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C8B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C8; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C8; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}


void fc1(const DataType N[OUT_IMAGE_SIZE_F1_IN],
		DataType P[OUT_IMAGE_SIZE_F1_OUT],
		const DataType FL[N_CLUSTERS][N_FFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//120 400

// Create local filter
const DataType M[OUT_IMAGE_SIZE_F1_OUT][OUT_IMAGE_SIZE_F1_IN] = {
		#include "f1_sym_filter.txt"
		};
const DataType F1BL[N_CLUSTERS][F1B] = {
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


// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		ReturnType Ps[OUT_IMAGE_SIZE_F1_OUT],
		DataType M1[C6B * FILTER_SIZE * FILTER_SIZE * C5B],
		DataType M2[C7B * FILTER_SIZE * FILTER_SIZE * C6B],
		DataType M3[C8B * FILTER_SIZE * FILTER_SIZE * C7B]) {

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp
//#pragma HLS INTERFACE mode=axis port=N_c1


	// Create local LUTs

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
	const ReturnType CODEBOOK[512] = {
					#include "centroid_lut.txt"
			 };

	// create output buffer
	DataType local_relu_1[OUT_IMAGE_SIZE_C1][OUT_IMAGE_SIZE_C1][C1B];

	// do first conv2D, return the buffer for next operations
	conv2D_c1(N_c1, local_relu_1, CL, AL);
	ReLU1(local_relu_1, RL);
	cout << "local_relu_1[0][0][0] = " << local_relu_1[0][0][0] << endl;

	// do second conv2D, return the buffer for next operations
	DataType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C2B];
	conv2D_c2(local_relu_1, local_conv_2, CL, AL);

	ReLU2(local_conv_2,  RL);
	cout << "local_conv_2[0][0][0] = " << local_conv_2[0][0][0] << endl;

	DataType local_conv_3[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C3B];
	conv2D_c3(local_conv_2, local_conv_3, CL, AL);
	ReLU3(local_conv_3,  RL);
	cout << "local_conv_3[0][0][0] = " << local_conv_3[0][0][0] << endl;

	DataType local_conv_4[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C4B];
	conv2D_c4(local_conv_3, local_conv_4, CL, AL);
	ReLU4(local_conv_4,  RL);
	cout << "local_conv_4[0][0][0] = " << local_conv_4[0][0][0] << endl;

	DataType local_conv_5[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C5B];
	conv2D_c5(local_conv_4, local_conv_5, CL, AL);
	ReLU5(local_conv_5, RL);
	cout << "local_conv_5[0][0][0] = " << local_conv_5[0][0][0] << endl;

	DataType local_conv_6[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C6B];
	//cout << "Initial convolution input = " << N_c1[0][0][0] << endl;
	conv2D_c6(local_conv_5, local_conv_6, M1, CL, AL);
	cout << "local_conv_6[0][0][0] = " << local_conv_6[0][0][0] << endl;
	ReLU6(local_conv_6,  RL);
	cout << "local_conv_6[0][0][0] after ReLU = " << local_conv_6[0][0][0] << endl;



	DataType local_conv_7[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C7B];
	conv2D_c7(local_conv_6, local_conv_7, M2, CL, AL);
	ReLU7(local_conv_7,  RL);
	cout << "local_conv_7[0][0][0] = " << local_conv_7[0][0][0] << endl;
	for (int l = 0; l < C7B; l++){     // image channels
		//for (int i; i <  OUT_IMAGE_SIZE_C7; i++){
			//for (int j; j <  OUT_IMAGE_SIZE_C7; j++){
				cout << "Symbol:" << local_conv_7[0][0][l] << " ";
			//}
		//}
	}
	cout <<  endl;

	// do second conv2D, return the buffer for next operations
	DataType local_conv_8[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C8B];
	//cout << "Initial convolution input = " << N_c1[0][0][0] << endl;
	conv2D_c8(local_conv_7, local_conv_8, M3, CL, AL);
	cout << "local_conv_8[0][0][0] = " << local_conv_8[0][0][0] << endl;
	ReLU8(local_conv_8,  RL);
	cout << "local_conv_8[0][0][0] after ReLU = " << local_conv_8[0][0][0] << endl;

	DataType local_fc[OUT_IMAGE_SIZE_F1_IN];
//	write_output:
	for (int l = 0; l < C8B; l++){     // image channels
				// actual multiply and add
				//local_fc[l] = local_conv_8[0][0][l];
				local_fc[l] = local_conv_7[0][0][l];
				cout << "Symbol: " << local_fc[l] << " ";

	}
	DataType local_fc_out[OUT_IMAGE_SIZE_F1_OUT];
	fc1(local_fc, local_fc_out, FL, AL);
	cout <<  endl;

	for (int l = 0; l < OUT_IMAGE_SIZE_F1_OUT; l++){     // image channels
					cout << "Symbol Final: " << local_fc_out[l] << " ";

		}

	cout <<  endl;

	for (int i = 0; i < OUT_IMAGE_SIZE_F1_OUT; i++){
		Ps[i] = CODEBOOK[local_fc_out[i]];
		cout << "Symbol: " << local_fc_out[i] << "Code: " << Ps[i] << endl;
	}
}
