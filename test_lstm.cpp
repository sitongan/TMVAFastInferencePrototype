#include "lstm_defaults.hxx"
#include "lstm_initial_bias.hxx"
#include "lstm_peepholes.hxx"
#include "lstm_batchwise.hxx"
#include "lstm_bidirectional.hxx"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <stdlib.h>
#include <time.h>

int main() {

   std::cout << std::endl << "Test 1: lstm defaults" << std::endl;
   float input1[6];
   std::iota(input1, input1 + 6, 1.);
   auto t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_lstm_defaults::infer(input1);
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y1 = TMVA_SOFIE_lstm_defaults::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y1[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.09524119, 0.09524119, 0.09524119
   // 0.32869044, 0.32869044, 0.32869044
   // 0.60042989, 0.60042989, 0.60042989
   std::cout << "y_h" << std::endl;
   float * y_h1 = TMVA_SOFIE_lstm_defaults::tensor_Yh;
   for (size_t i = 0; i < 3; i++)
      std::cout << y_h1[i] << "  ";
   std::cout << std::endl;
   // true y_h
   // 0.60042989, 0.60042989, 0.60042989

   std::cout << std::endl << "Test 2: lstm initial_bias" << std::endl;
   float input2[9];
   std::iota(input2, input2 + 9, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_lstm_initial_bias::infer(input2);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y2 = TMVA_SOFIE_lstm_initial_bias::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y2[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.25606444, 0.25606444, 0.25606444, 0.25606444
   // 0.68688357, 0.68688357, 0.68688357, 0.68688357
   // 0.90747154, 0.90747154, 0.90747154, 0.90747154
   std::cout << "y_h" << std::endl;
   float * y_h2 = TMVA_SOFIE_lstm_initial_bias::tensor_Yh;
   for (size_t i = 0; i < 4; i++)
      std::cout << y_h2[i] << "  ";
   std::cout << std::endl;
   // true y_h
   // 0.90747154, 0.90747154, 0.90747154, 0.90747154

   std::cout << std::endl << "Test 3: lstm peepholes" << std::endl;
   float input3[8];
   std::iota(input3, input3 + 8, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_lstm_peepholes::infer(input3);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y3 = TMVA_SOFIE_lstm_peepholes::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y3[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.37506911, 0.37506911, 0.37506911
   // 0.6801309,  0.6801309,  0.6801309
   std::cout << "y_h" << std::endl;
   float * y_h3 = TMVA_SOFIE_lstm_peepholes::tensor_Yh;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y_h3[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.37506911, 0.37506911, 0.37506911
   // 0.6801309,  0.6801309,  0.6801309

   std::cout << std::endl << "Test 4: lstm batchwise" << std::endl;
   float input4[6];
   std::iota(input4, input4 + 6, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_lstm_batchwise::infer(input4);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y4 = TMVA_SOFIE_lstm_batchwise::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 7; j++)
         std::cout << y4[i * 7 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258,
   // 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319,
   // 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899
   std::cout << "y_h" << std::endl;
   float * y_h4 = TMVA_SOFIE_lstm_batchwise::tensor_Yh;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 7; j++)
         std::cout << y_h4[i * 7 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258,
   // 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319,
   // 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899

   std::cout << std::endl << "Test 5: lstm bidirectional" << std::endl;
   float input5[6];
   std::iota(input5, input5 + 6, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_lstm_bidirectional::infer(input5);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y5 = TMVA_SOFIE_lstm_bidirectional::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y5[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.0952, 0.0952, 0.0952,
   // 0.4041, 0.4041, 0.4041,
   // 0.3287, 0.3287, 0.3287,
   // 0.4927, 0.4927, 0.4927,
   // 0.6004, 0.6004, 0.6004,
   // 0.4032, 0.4032, 0.4032
   std::cout << "y_h" << std::endl;
   float * y_h5 = TMVA_SOFIE_lstm_bidirectional::tensor_Yh;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y_h5[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.6004, 0.6004, 0.6004,
   // 0.4041, 0.4041, 0.4041
   std::cout << "y_c" << std::endl;
   float * y_c5 = TMVA_SOFIE_lstm_bidirectional::tensor_Yc;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++)
         std::cout << y_c5[i * 3 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_c
   // 1.0493, 1.0493, 1.0493,
   // 0.7970, 0.7970, 0.7970
}
