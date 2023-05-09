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

typedef float DataType;

using namespace std;

void cnn_forward(const DataType N_c1[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL],
		DataType Ps[OUT_IMAGE_SIZE_F1_OUT],
		DataType M1[C6B * FILTER_SIZE * FILTER_SIZE * C5B],
		DataType M2[C7B * FILTER_SIZE * FILTER_SIZE * C6B],
		DataType M3[C8B * FILTER_SIZE * FILTER_SIZE * C7B]);

//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL] = {
    		#include "img_std_label_2.txt"
    		};

    //DataType *inp_image = (DataType *)malloc( OUT_IMAGE_SIZE_C1 * OUT_IMAGE_SIZE_C1 * C1B * sizeof(DataType));
    DataType *c1_return = (DataType *)malloc( OUT_IMAGE_SIZE_F1_OUT * sizeof(DataType));
    

    //cout << "Start calling the conv1 HW function" << endl;

    ifstream ff1("/home/swarnava/Projects/Work/aiml/cvpr2023/code/CIFAR_VGG_FPGA/vgg_std/c6f.txt");
    DataType *M1;
    M1 = (DataType *)malloc(C6B * FILTER_SIZE * FILTER_SIZE * C5B * sizeof(DataType));
	if(ff1.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C6B * FILTER_SIZE * FILTER_SIZE * C5B) ; i++)
		{
			ff1 >> M1[i];
		}
		ff1.close();
	}
    //cout << "M1[0] = " << M1[0] << endl;


    ifstream ff2("/home/swarnava/Projects/Work/aiml/cvpr2023/code/CIFAR_VGG_FPGA/vgg_std/c7f.txt");
    DataType *M2;

    M2 = (DataType *)malloc( C7B * FILTER_SIZE * FILTER_SIZE * C6B * sizeof(DataType));
    if(ff2.is_open())
    {
    	//cout << "can open the filter file" << endl;

    	for (int i=0; i< (C7B * FILTER_SIZE * FILTER_SIZE * C6B) ; i++)
    	{
    		ff2 >> M2[i];
    	}
    	ff2.close();
    }
    //cout << "M2[0] = " << M2[0] << endl;


	ifstream ff3("/home/swarnava/Projects/Work/aiml/cvpr2023/code/CIFAR_VGG_FPGA/vgg_std/c8f.txt");
	DataType *M3;

	M3 = (DataType *)malloc( C8B * FILTER_SIZE * FILTER_SIZE * C7B * sizeof(DataType));
	if(ff3.is_open())
	{
		cout << "can open the filter file" << endl;

		for (int i=0; i< (C8B * FILTER_SIZE * FILTER_SIZE * C7B) ; i++)
		{
			ff3 >> M3[i];
            //cout << M3[i];
		}
		//cout << endl;
		ff3.close();
		//cout << " Filter file loaded M3 " << M3[0] << endl;

	}


 //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
 cnn_forward(IMG, c1_return, M1, M2, M3);

 //cout << "After calling the conv1 HW function" << endl;

 ofstream outfile ("output_std.txt");
 if (outfile.is_open())
   {
	 cout << "Printing logits for CIFAR-10: " << endl;
	 for(int count = 0; count < OUT_IMAGE_SIZE_F1_OUT; count ++){
    	 outfile << c1_return[count] << "," << endl;
    	 cout <<  "Logit:" << c1_return[count] << endl;
     }

   }
   //else cout << "Unable to open output file";
   outfile.close();
 //cout << "After saving conv1 output" << endl;



 free(c1_return);
 free(M1);
 free(M2);
 free(M3);

 cout << "Functionality pass" << endl;
  
 return 0;
}
