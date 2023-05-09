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
		DataType Ps[FOUT],
		DataType ML31C1[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML31C2[C3B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C1[C4B * FILTER_SIZE * FILTER_SIZE * C3B],
		DataType ML40C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C1[C4B * FILTER_SIZE * FILTER_SIZE * C4B],
		DataType ML41C2[C4B * FILTER_SIZE * FILTER_SIZE * C4B]);

//main function used to test the functionality of the kernel.
int main()
{
    const DataType IMG[INP_IMAGE_SIZE][INP_IMAGE_SIZE][INP_IMAGE_CHANNEL] = {
    		#include "imgnet_img_std_label_53.txt"
    		};

    DataType *c1_return = (DataType *)malloc( FOUT * sizeof(DataType));

    
    ifstream ff1("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c12f.txt");
    DataType *ML31C1;
    ML31C1 = (DataType *)malloc(C3B * FILTER_SIZE * FILTER_SIZE * C3B * sizeof(DataType));
	if(ff1.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C3B * FILTER_SIZE * FILTER_SIZE * C3B) ; i++)
		{
			ff1 >> ML31C1[i];
		}
		ff1.close();
	}


   ifstream ff2("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c13f.txt");
	DataType *ML31C2;
	ML31C2 = (DataType *)malloc(C3B * FILTER_SIZE * FILTER_SIZE * C3B * sizeof(DataType));
	if(ff2.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C3B * FILTER_SIZE * FILTER_SIZE * C3B) ; i++)
		{
			ff2 >> ML31C2[i];
		}
		ff2.close();
	}

   ifstream ff3("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c14f.txt");
	DataType *ML40C1;
	ML40C1 = (DataType *)malloc(C4B * FILTER_SIZE * FILTER_SIZE * C3B * sizeof(DataType));
	if(ff3.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C4B * FILTER_SIZE * FILTER_SIZE * C3B) ; i++)
		{
			ff3 >> ML40C1[i];
		}
		ff3.close();
	}

   ifstream ff4("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c15f.txt");
	DataType *ML40C2;
	ML40C2 = (DataType *)malloc(C4B * FILTER_SIZE * FILTER_SIZE * C4B * sizeof(DataType));
	if(ff4.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C4B * FILTER_SIZE * FILTER_SIZE * C4B) ; i++)
		{
			ff4 >> ML40C2[i];
		}
		ff4.close();
	}

   ifstream ff5("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c16f.txt");
	DataType *ML41C1;
	ML41C1 = (DataType *)malloc(C4B * FILTER_SIZE * FILTER_SIZE * C4B * sizeof(DataType));
	if(ff5.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C4B * FILTER_SIZE * FILTER_SIZE * C4B) ; i++)
		{
			ff5 >> ML41C1[i];
		}
		ff5.close();
	}
   ifstream ff6("/home/swarnava/Projects/Work/aiml/cvpr2023/code/IMGNET_RESNET/rimg_std/c17f.txt");
	DataType *ML41C2;
	ML41C2 = (DataType *)malloc(C4B * FILTER_SIZE * FILTER_SIZE * C4B * sizeof(DataType));
	if(ff6.is_open())
	{
		//cout << "can open the filter file" << endl;

		for (int i=0; i< (C4B * FILTER_SIZE * FILTER_SIZE * C4B) ; i++)
		{
			ff6 >> ML41C2[i];
		}
		ff6.close();
	}


 //call the "conv1" function using the "inp_image" argument, it returns the output in the "out_image" array
 cnn_forward(IMG, c1_return, ML31C1,ML31C2,ML40C1,ML40C2,ML41C1,ML41C2);

 //cout << "After calling the conv1 HW function" << endl;

 ofstream outfile ("output_std.txt");
 if (outfile.is_open())
   {
	 cout << "Printing logits for ImageNet: " << endl;
	 for(int count = 0; count < FOUT; count ++){
    	 outfile << c1_return[count] << "," << endl;
    	 //cout <<  "Logit:" << c1_return[count] << endl;
     }

   }
   else cout << "Unable to open output file";
       outfile.close();



 free(c1_return);
 free(ML31C1);
 free(ML31C2);
 free(ML40C1);
 free(ML40C2);
 free(ML41C1);
 free(ML41C2);

 cout << "Functionality pass" << endl;
  
 return 0;
}
