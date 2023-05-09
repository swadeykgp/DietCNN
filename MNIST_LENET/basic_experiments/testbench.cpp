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
		DataType *Ps);


//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG=232;


    DataType *softmax_out = (DataType *)malloc(sizeof(DataType));
    
    cout << "Start calling the test function" << endl;


  //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
  cnn_forward(IMG, softmax_out);

  cout << "After calling the test function" << *softmax_out << endl;


  free(softmax_out);

  cout << "Functionality pass" <<  endl;
  
  return 0;
}
