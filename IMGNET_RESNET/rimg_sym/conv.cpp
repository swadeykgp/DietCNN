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

typedef ushort DataType;
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
const DataType M[C1B][FILTER_SIZE_C1][FILTER_SIZE_C1][INP_IMAGE_CHANNEL] = {
		                                                                         #include "c1f.txt"
		     };


int tmp = 0;
const int tmp_symbol_num = INP_IMAGE_CHANNEL*FILTER_SIZE_C1*FILTER_SIZE_C1;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

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
                P[(i/STRIDE_C1)][(j/STRIDE_C1)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d c1



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
		DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c2f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C1B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU2(DataType P[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C2; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C2; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c3(DataType N[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B],
		DataType P[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C1B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c3f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C1B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C2 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void conv2D_c4(DataType N[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C1B],
		DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C1B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c4f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C1B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C3 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU3(DataType P[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C1B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C1B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C4; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C4; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c5(DataType N[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C1B],
		DataType P[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C1B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C1B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c5f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C1B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C1B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C4 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2



void conv2D_c6(DataType N[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C1B],
		DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C2B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C1B] = {
		                                                                         #include "c6f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C2B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C5 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU4(DataType P[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C2B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C6; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C6; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c7(DataType N[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C2B],
		DataType P[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C2B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c7f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C2B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C6 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2


void conv2D_c8(DataType N[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C2B],
		DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C2B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c8f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C2B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C7 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C7 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU5(DataType P[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C2B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C2B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C8; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C8; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c9(DataType N[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C2B],
		DataType P[OUT_IMAGE_SIZE_C9][OUT_IMAGE_SIZE_C9][C2B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C2B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c9f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C2B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C2B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C8 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C8 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2


void conv2D_c10(DataType N[OUT_IMAGE_SIZE_C9][OUT_IMAGE_SIZE_C9][C2B],
		DataType P[OUT_IMAGE_SIZE_C10][OUT_IMAGE_SIZE_C10][C3B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C2B] = {
		                                                                         #include "c10f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C3B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C9 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C9 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU6(DataType P[OUT_IMAGE_SIZE_C10][OUT_IMAGE_SIZE_C10][C3B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C10; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C10; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}


void conv2D_c11(DataType N[OUT_IMAGE_SIZE_C10][OUT_IMAGE_SIZE_C10][C3B],
		DataType P[OUT_IMAGE_SIZE_C11][OUT_IMAGE_SIZE_C11][C3B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {

// Create local filter
// This one is full implementation with channel
const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C3B] = {
		                                                                         #include "c11f.txt"
		     };
int tmp = 0;
const int tmp_symbol_num = C3B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C10 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C10 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void conv2D_c12(DataType N[OUT_IMAGE_SIZE_C11][OUT_IMAGE_SIZE_C11][C3B],
		DataType P[OUT_IMAGE_SIZE_C12][OUT_IMAGE_SIZE_C12][C3B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS], DataType M[C3B * FILTER_SIZE * FILTER_SIZE * C3B]) {

//// Create local filter
//// This one is full implementation with channel
//const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C3B] = {
//		                                                                         #include "c12f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C3B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C11 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C11 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C3B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU7(DataType P[OUT_IMAGE_SIZE_C12][OUT_IMAGE_SIZE_C12][C3B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C3B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C12; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C12; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c13(DataType N[OUT_IMAGE_SIZE_C12][OUT_IMAGE_SIZE_C12][C3B],
		DataType P[OUT_IMAGE_SIZE_C13][OUT_IMAGE_SIZE_C13][C3B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS], DataType M[C3B * FILTER_SIZE * FILTER_SIZE * C3B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C3B][FILTER_SIZE][FILTER_SIZE][C3B] = {
//		                                                                         #include "c13f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C3B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C3B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C12 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C12 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C3B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2


void conv2D_c14(DataType N[OUT_IMAGE_SIZE_C13][OUT_IMAGE_SIZE_C13][C3B],
		DataType P[OUT_IMAGE_SIZE_C14][OUT_IMAGE_SIZE_C14][C4B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS], DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C3B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C3B] = {
//		                                                                         #include "c14f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C4B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C13 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C13 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C3B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void ReLU8(DataType P[OUT_IMAGE_SIZE_C14][OUT_IMAGE_SIZE_C14][C4B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C14; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C14; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}

void conv2D_c15(DataType N[OUT_IMAGE_SIZE_C14][OUT_IMAGE_SIZE_C14][C4B],
		DataType P[OUT_IMAGE_SIZE_C15][OUT_IMAGE_SIZE_C15][C4B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS],DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C4B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C4B] = {
//		                                                                         #include "c15f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C4B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C14 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C14 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2


void conv2D_c16(DataType N[OUT_IMAGE_SIZE_C15][OUT_IMAGE_SIZE_C15][C4B],
		DataType P[OUT_IMAGE_SIZE_C16][OUT_IMAGE_SIZE_C16][C4B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS],DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C4B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C4B] = {
//		                                                                         #include "c16f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C4B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C15 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C15 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[M[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2
void ReLU9(DataType P[OUT_IMAGE_SIZE_C16][OUT_IMAGE_SIZE_C16][C4B], const DataType RL[N_CLUSTERS]) {
	int ii = 0;
	for (int k = 0; k < C4B; k++){     // image channels

		for (int i = 0; i < OUT_IMAGE_SIZE_C16; i++){              // rows

			for (int j = 0; j < OUT_IMAGE_SIZE_C16; j++){          // columns

				P[i][j][k] = RL[P[i][j][k]];
			}
		}
	}
}
void conv2D_c17(DataType N[OUT_IMAGE_SIZE_C16][OUT_IMAGE_SIZE_C16][C4B],
		DataType P[OUT_IMAGE_SIZE_C17][OUT_IMAGE_SIZE_C17][C4B],
		const DataType CL[N_CLUSTERS][N_CFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS],DataType M[C4B * FILTER_SIZE * FILTER_SIZE * C4B]) {

// Create local filter
// This one is full implementation with channel
//const DataType M[C4B][FILTER_SIZE][FILTER_SIZE][C4B] = {
//		                                                                         #include "c17f.txt"
//		     };
int tmp = 0;
const int tmp_symbol_num = C4B*FILTER_SIZE*FILTER_SIZE;
DataType tmp_conv_sym[tmp_symbol_num];

    for (int o = 0; o < C4B; o++){ // output filter

        for (int i = 0; i < OUT_IMAGE_SIZE_C16 - FILTER_SIZE; i=i+STRIDE_ALL){              // rows

            for (int j = 0; j < OUT_IMAGE_SIZE_C16 - FILTER_SIZE; j=j+STRIDE_ALL){          // columns

            	tmp = 4096;
            	DataType tmp_conv_sym[tmp_symbol_num];
                for (int l = 0; l < C4B; l++){ // image channels

                    for (int m = 0; m < FILTER_SIZE; m++){ // kernel rows

                        for (int n = 0; n < FILTER_SIZE; n++){ // kernel columns

                            // actual multiply and add replaced by table lookup
                        	tmp_conv_sym[l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n] = CL[N[i+m][j+n][l]][M[o*FILTER_SIZE*FILTER_SIZE*C4B + l*FILTER_SIZE*FILTER_SIZE + m*FILTER_SIZE + n]];
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
                P[(i/STRIDE_ALL)][(j/STRIDE_ALL)][o] =  tmp;
                tmp = 0;
            }
        } // end of one output channel , bias should have been added here. But we add for each out pixel
    }
} // conv2d 2

void fc1(const DataType N[F1B],
		DataType P[FOUT],
		const DataType FL[N_CLUSTERS][N_FFILTERS],
		const DataType AL[N_CLUSTERS][N_CLUSTERS]) {
//120 400

// Create local filter
const DataType M[FOUT][F1B] = {
		#include "ff.txt"
		};
   int tmp = 0;
   const int tmp_symbol_num = F1B;
   DataType tmp_sym[tmp_symbol_num];

    for (int i = 0; i < FOUT; i++){     // image rows

        tmp = 0;
        for (int j = 0; j < F1B; j++){ //  image columns
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
		P[i] =  tmp;
		tmp = 0;
    }
}


// This is the top function

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		ReturnType Ps[FOUT],
		DataType ML31C1[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML31C2[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C1[C4B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C1[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B]) {

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

//	// do second conv2D, return the buffer for next operations
//	DataType local_conv_2[OUT_IMAGE_SIZE_C2][OUT_IMAGE_SIZE_C2][C1B];
//	conv2D_c2(local_relu_1, local_conv_2, CL, AL);
//
//	ReLU2(local_conv_2,  RL);
//
//	DataType local_conv_3[OUT_IMAGE_SIZE_C3][OUT_IMAGE_SIZE_C3][C1B];
//	conv2D_c3(local_conv_2, local_conv_3, CL, AL);
//
//	DataType local_conv_4[OUT_IMAGE_SIZE_C4][OUT_IMAGE_SIZE_C4][C1B];
//	conv2D_c4(local_conv_3, local_conv_4, CL, AL);
//	ReLU3(local_conv_4,  RL);
//
//	DataType local_conv_5[OUT_IMAGE_SIZE_C5][OUT_IMAGE_SIZE_C5][C1B];
//	conv2D_c5(local_conv_4, local_conv_5, CL, AL);
//	DataType local_conv_6[OUT_IMAGE_SIZE_C6][OUT_IMAGE_SIZE_C6][C2B];
//	conv2D_c6(local_conv_5, local_conv_6, CL, AL);
//	ReLU4(local_conv_6, RL);
//	DataType local_conv_7[OUT_IMAGE_SIZE_C7][OUT_IMAGE_SIZE_C7][C2B];
//	conv2D_c7(local_conv_6, local_conv_7,  CL, AL);
//	DataType local_conv_8[OUT_IMAGE_SIZE_C8][OUT_IMAGE_SIZE_C8][C2B];
//	conv2D_c8(local_conv_7, local_conv_8,  CL, AL);
//	ReLU5(local_conv_8,  RL);
//	DataType local_conv_9[OUT_IMAGE_SIZE_C9][OUT_IMAGE_SIZE_C9][C2B];
//	conv2D_c9(local_conv_8, local_conv_9,  CL, AL);


	DataType local_conv_10[OUT_IMAGE_SIZE_C10][OUT_IMAGE_SIZE_C10][C3B];
//	conv2D_c10(local_conv_9, local_conv_10,  CL, AL);
	ReLU6(local_conv_10,  RL);
	DataType local_conv_11[OUT_IMAGE_SIZE_C11][OUT_IMAGE_SIZE_C11][C3B];
	conv2D_c11(local_conv_10, local_conv_11,  CL, AL);
	DataType local_conv_12[OUT_IMAGE_SIZE_C12][OUT_IMAGE_SIZE_C12][C3B];
	conv2D_c12(local_conv_11, local_conv_12,  CL, AL, ML31C1);
	ReLU7(local_conv_12,  RL);
	DataType local_conv_13[OUT_IMAGE_SIZE_C13][OUT_IMAGE_SIZE_C13][C3B];
	conv2D_c13(local_conv_12, local_conv_13,  CL, AL, ML31C2);

	DataType local_conv_14[OUT_IMAGE_SIZE_C14][OUT_IMAGE_SIZE_C14][C4B];
	conv2D_c14(local_conv_13, local_conv_14,  CL, AL, ML40C1);
	ReLU8(local_conv_14,  RL);
	DataType local_conv_15[OUT_IMAGE_SIZE_C15][OUT_IMAGE_SIZE_C15][C4B];
	conv2D_c15(local_conv_14, local_conv_15,  CL, AL, ML40C2);
	DataType local_conv_16[OUT_IMAGE_SIZE_C16][OUT_IMAGE_SIZE_C16][C4B];
	conv2D_c16(local_conv_15, local_conv_16,  CL, AL, ML41C1);
	ReLU9(local_conv_16,  RL);
	DataType local_conv_17[OUT_IMAGE_SIZE_C17][OUT_IMAGE_SIZE_C17][C4B];
	conv2D_c17(local_conv_16, local_conv_17,  CL, AL, ML41C2);
	DataType local_fc_in[F1B];
	for (int l = 0; l < C4B; l++){     // image channels
		local_fc_in[l] = local_conv_17[0][0][l];
	}
	DataType local_fc_out[FOUT];
	fc1(local_fc_in, local_fc_out, FL, AL);

	for (int i = 0; i < FOUT; i++){
		Ps[i] = (ReturnType)CODEBOOK[local_fc_out[i]];
	}
}
