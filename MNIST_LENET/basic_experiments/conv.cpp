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


//typedef float DataType;
typedef int DataType;


using namespace std;


void cnn_forward(const DataType N_c1,
		DataType *Ps) {

	// First step is data copy in local buffer. That's how we code for FPGA implementations
	// see: https://github.com/Xilinx/Vitis_Accel_Examples/blob/2020.2/cpp_kernels/array_partition/src/matmul_partition.cpp

#pragma HLS INTERFACE mode=axis port=N_c1
#pragma HLS INTERFACE mode=axis port=Ps
	const DataType add_lut[512][512]= {
	    		#include "add_lut.txt"
	    		};

   	//*Ps = N_c1*23123456.00987453;
   	*Ps = add_lut[1][511];
	//*Ps = N_c1 >> 2;
	//*Ps = N_c1 + 2000;

}
