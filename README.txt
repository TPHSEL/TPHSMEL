//This folder contains 28 datasets.
//List of the datasets is in the paper.
//This folder also contains the code file TPHSMEL.cpp
//This folder also contains its executable file TPHSMEL.exe
//This code can run on linux terminal
//You can run executable file directly with following command

./TPHSMEL.exe name_of_the_dataset.txt prefix_of_output_file

//or you can create your own executable file with the following command

g++ -std=c++11 TPHSMEL.cpp -o TPHSMEL.exe

//It is clear from above command that you need to have software of c++11 installed.

//The output will consist of four files.

1. prefix_training_2_13.xls
2. prefix_test_2_13.xls
3. prefix_ensemble_2_13.xls
4. prefix_levels_2_13.xls

//It is clear from the names of the files that
//First file contains details of the trained models in each hierarchy
//Second file contains test results of each iteration (13 iterations) within ensemble.
//Third file aggregates the results of second file to provide final classification results.
//Forth file provides details of in which hierarchy the instance was classified

have fun!