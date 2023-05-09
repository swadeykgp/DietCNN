//#include <sys/types.h>
//#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <fstream>
#include "conv.h"
#include <ap_fixed.h>

//typedef float DataType;
typedef ap_fixed<16,6> DataType;
using namespace std;

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType Ps[OUT_SOFTMAX]);


//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL] = {
    		#include "img_std_label_7.txt"
    		};


    DataType *softmax_out = (DataType *)malloc(OUT_SOFTMAX * sizeof(DataType));
    
    cout << "Start calling the conv1 HW function" << endl;


  //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
  cnn_forward(IMG, softmax_out);

  cout << "After calling the conv1 HW function" << endl;

  DataType max = 0.0;
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
