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

typedef int DataType;
typedef float ReturnType;

using namespace std;

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		ReturnType Ps[OUT_SOFTMAX]);


//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL] = {
    		#include "img_sym_label_7.txt"
    		};

    ReturnType *softmax_out = (ReturnType *)malloc(OUT_SOFTMAX * sizeof(ReturnType));
    

  cout << "Start calling the conv1 HW function" << endl;


  //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
  cnn_forward(IMG, softmax_out);

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

  free(softmax_out);

  cout << "Functionality pass" << endl;
  
  return 0;
}
