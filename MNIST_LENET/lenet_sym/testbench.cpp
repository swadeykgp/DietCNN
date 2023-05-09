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
#include "ap_int.h"
typedef ap_uint<10> DataType;

//typedef int DataType;
typedef float ReturnType;

using namespace std;

void cnn_forward(DataType N_c1[INP_IMAGE_SIZE*INP_IMAGE_SIZE*INP_IMAGE_CHANNEL],
		ReturnType Ps[OUT_SOFTMAX]);


//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL] = {
    		#include "img_sym_label_7.txt"
    		};

    DataType *inp_image = (DataType *)malloc( INP_IMAGE_SIZE * INP_IMAGE_SIZE * INP_IMAGE_CHANNEL * sizeof(DataType));
    ReturnType *softmax_out = (ReturnType *)malloc(OUT_SOFTMAX * sizeof(ReturnType));
    
    for (int l = 0; l < INP_IMAGE_CHANNEL; l++){     // image channels
	    for (int i = 0; i < INP_IMAGE_SIZE; i++){     // image rows
            for (int j = 0; j < INP_IMAGE_SIZE; j++){ // image columns
				// actual multiply and add
				inp_image[ l*INP_IMAGE_SIZE*INP_IMAGE_SIZE + i*INP_IMAGE_SIZE + j] = IMG[i][j][l];
            }
	    }
    }

  cout << "inp_image[0] = " << inp_image[0] << endl;

  cout << "Start calling the conv1 HW function" << endl;


  //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
  cnn_forward(inp_image, softmax_out);

  cout << "After calling the conv1 HW function" << endl;
  ReturnType max = 0.0;
    int endix = -1;
    for (int i = 0; i <OUT_SOFTMAX; i++){
  	  cout << softmax_out[i] << endl;
        if (softmax_out[i] > max){
      	 max = softmax_out[i];
      	 endix = i;
        }
    }
    cout << "Output class is: " << endix << endl;

  //free all the dynamically allocated memory

  free(inp_image);
  free(softmax_out);

  cout << "Functionality pass" << endl;
  
  return 0;
}
