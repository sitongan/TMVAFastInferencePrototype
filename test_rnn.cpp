#include "rnn_defaults.hxx"
#include "rnn_seq_length.hxx"
#include "rnn_batchwise.hxx"
#include "rnn_bidirectional.hxx"
#include "rnn_bidirectional_batchwise.hxx"
#include "rnn_sequence.hxx"
#include "rnn_sequence_batchwise.hxx"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <stdlib.h>
#include <time.h>

int main() {

   std::cout << std::endl << "Test 1: rnn defaults" << std::endl;
   float input[9];
   std::iota(input, input + 9, 1.);
   auto t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_defaults::infer(input);
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y = TMVA_SOFIE_rnn_defaults::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 5; j++)
         std::cout << y[i * 5 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.06988589, 0.06988589, 0.06988589, 0.06988589, 0.06988589,
   //   0.16205294, 0.16205294, 0.16205294, 0.16205294, 0.16205294,
   //  0.25251999, 0.25251999, 0.25251999, 0.25251999, 0.25251999
   std::cout << "y_h" << std::endl;
   float * y_h = TMVA_SOFIE_rnn_defaults::tensor_Yh;
   for (size_t i = 0; i < 5; i++)
      std::cout << y_h[i] << "  ";
   std::cout << std::endl;
   // true y_h
   // 0.25251999, 0.25251999, 0.25251999, 0.25251999, 0.25251999


   std::cout << std::endl << "Test 2: rnn seq_length" << std::endl;
   float input2[18];
   std::iota(input2, input2 + 18, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_seq_length::infer(input2);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   std::cout << "y" << std::endl;
   float* y2 = TMVA_SOFIE_rnn_seq_length::tensor_Y;
   for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 5; j++)
         std::cout << y2[i * 5 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.17032367, 0.17032367, 0.17032367, 0.17032367, 0.17032367,
   // 0.33814806, 0.33814806, 0.33814806, 0.33814806, 0.33814806,
   // 0.48690841, 0.48690841, 0.48690841, 0.48690841, 0.48690841,
   // 0.622473,   0.622473,   0.622473,   0.622473,   0.622473,
   // 0.72863662, 0.72863662, 0.72863662, 0.72863662, 0.72863662,
   // 0.80780905, 0.80780905, 0.80780905, 0.80780905, 0.80780905
   float* y_h2 = TMVA_SOFIE_rnn_seq_length::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 5; j++)
         std::cout << y_h2[i * 5 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.622473,   0.622473,   0.622473,   0.622473,   0.622473,
   // 0.72863662, 0.72863662, 0.72863662, 0.72863662, 0.72863662,
   // 0.80780905, 0.80780905, 0.80780905, 0.80780905, 0.80780905


   std::cout << std::endl << "Test 3: rnn batchwise" << std::endl;
   float input3[6];
   std::iota(input3, input3 + 6, 1.);
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_batchwise::infer(input3);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float* y3 = TMVA_SOFIE_rnn_batchwise::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y3[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // 0.14888504, 0.14888504, 0.14888504, 0.14888504,
   // 0.33637556, 0.33637556, 0.33637556, 0.33637556,
   // 0.50052023, 0.50052023, 0.50052023, 0.50052023
   float *y_h3 = TMVA_SOFIE_rnn_batchwise::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y_h3[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.14888504, 0.14888504, 0.14888504, 0.14888504,
   // 0.33637556, 0.33637556, 0.33637556, 0.33637556,
   // 0.50052023, 0.50052023, 0.50052023, 0.50052023


   std::cout << std::endl << "Test 4: rnn bidirectional" << std::endl;
   float input4[18] = {
      0.,    0.01, 0.02, 0.03, 0.04, 0.05,
      0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
      0.12, 0.13, 0.14, 0.15, 0.16, 0.17};
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_bidirectional::infer(input4);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float *y4 = TMVA_SOFIE_rnn_bidirectional::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 18; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y4[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // -0.1680, -0.9967,  0.9200,  0.9520,
   // -0.0252,  0.9499,  0.2707,  0.9354,
   // 0.9207,  -0.9991,  0.4916, -0.9971,
   // 0.9570,   0.9907,  0.8348, -0.8432,
   // 0.9337,   0.9806,  0.4609, -0.8339,
   // 0.9651,   0.9867, -0.1637, -0.9962,
   // 0.9723,  -1.0000,  0.6775, -0.8878,
   // -0.4064, -1.0000, -0.3654, -0.9542,
   // 0.6047,  -0.9999, -0.4339,  0.0454,
   // 0.5554,   0.9138, -0.3105,  0.2289,
   // 0.3086,   0.9752,  0.0683, -0.4540,
   // -0.7255,  0.6310, -0.9871, -0.7722,
   // 0.6853,  -1.0000, -0.5145, -0.2746,
   // 0.8194,  -0.5044,  0.7796,  0.8733,
   // 0.9203,  -0.9999,  0.8698,  0.9291,
   // 1.0000,   0.9964,  0.9936, -0.9419,
   // 0.9995,   0.9907,  0.4186, -0.9974,
   // 0.9966,  -0.5491, -0.9923, -0.5074
   float* y_h4 = TMVA_SOFIE_rnn_bidirectional::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y_h4[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.68528736, -0.99995285, -0.51453173, -0.27458954,
   // 0.81935215, -0.5043667,   0.7795707,   0.8733188,
   // 0.92034006, -0.999916,    0.8697902,   0.9291247,
   // 0.9570278,   0.990718,    0.83482444, -0.8432297,
   // 0.9336523,   0.98059773,  0.46091712, -0.83394974,
   // 0.9651197,   0.98665035, -0.16370827, -0.99619436


   std::cout << std::endl << "Test 5: rnn bidirectional batchwise" << std::endl;
   float input5[18] = {
      0,    0.01, 0.06, 0.07, 0.12, 0.13,
      0.02, 0.03, 0.08, 0.09, 0.14, 0.15,
      0.04, 0.05, 0.1,  0.11, 0.16, 0.17};
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_bidirectional_batchwise::infer(input5);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float *y5 = TMVA_SOFIE_rnn_bidirectional_batchwise::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 18; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y5[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // -0.168,  -0.9967,  0.92,     0.952,
   // 0.957,   0.9907,  0.8348,  -0.8432,
   // 0.9723, -1,       0.6775,  -0.8878,
   // 0.5554,  0.9138, -0.3105,   0.2289,
   // 0.6853, -1,      -0.5145,  -0.2746,
   // 1,       0.9964,  0.9936,  -0.9419,
   // -0.0252,  0.9499,  0.2707,   0.9354,
   // 0.9337,  0.9806,  0.4609,  -0.8339,
   // -0.4064, -1,       -0.3654, -0.9542,
   // 0.3086,  0.9752,  0.0683,  -0.454,
   // 0.8194, -0.5044,  0.7796,   0.8733,
   // 0.9995,  0.9907,  0.4186,  -0.9974,
   // 0.9207, -0.9991,  0.4916,  -0.9971,
   // 0.9651,  0.9867, -0.1637,  -0.9962,
   // 0.6047, -0.9999, -0.4339,   0.0454,
   // -0.7255,  0.631,  -0.9871,  -0.7722,
   // 0.9203, -0.9999,  0.8698,   0.9291,
   // 0.9966, -0.5491, -0.9923,  -0.5074
   float* y_h5 = TMVA_SOFIE_rnn_bidirectional_batchwise::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 6; i++) {
      for (size_t j = 0; j < 4; j++)
         std::cout << y_h5[i * 4 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // 0.685287, -0.999953, -0.514532, -0.27459,
   // 0.957028,  0.990718,  0.834824, -0.84323,
   // 0.819352, -0.504367,  0.779571,  0.873319,
   // 0.933652,  0.980598,  0.460917, -0.83395,
   // 0.92034,  -0.999916,  0.86979,   0.929125,
   // 0.96512,   0.98665,  -0.163708, -0.996194


   std::cout << std::endl << "Test 6: rnn sequence" << std::endl;
   float input6[45] = {
       0.01,  -0.01,   0.08,   0.09,    0.001,
       0.09,   -0.7,   -0.35,   0.0,     0.001,
       0.16,  -0.19,   0.003,  0.,      0.0001,
       0.05,  -0.09,   0.013,  0.5,     0.005,
       .2,    -0.05,   .062,  -0.04,   -0.04,
       0.,     0.,     0.,     0.,      0.,
       0.06,   0.087,  0.01,   0.3,    -0.001,
       0.,     0.,     0.,     0.,      0.,
       0.,     0.,     0.,     0.,      0.};
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_sequence::infer(input6);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float *y6 = TMVA_SOFIE_rnn_sequence::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 9; i++) {
      for (size_t j = 0; j < 6; j++)
         std::cout << y6[i * 6 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // -0.0160, -0.1818, -0.0401, -0.0794,  0.1761,  0.0137,
   // -0.1869,  0.8827, -0.6948, -0.2732,  0.4479,  0.9408,
   // 0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097,
   // -0.4409, -0.5119, -0.1651,  0.0995,  0.8556, -0.4281,
   // -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
   // -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
   float* y_h6 = TMVA_SOFIE_rnn_sequence::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 6; j++)
         std::cout << y_h6[i * 6 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
   // -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
   // 0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097


   std::cout << std::endl << "Test 7: rnn sequence batchwise" << std::endl;
   float input7[45] = {
       0.01,  -0.01,   0.08,   0.09,    0.001,
       0.05,  -0.09,   0.013,  0.5,     0.005,
       0.06,   0.087,  0.01,   0.3,    -0.001,
       0.09,   -0.7,   -0.35,   0.0,     0.001,
       .2,    -0.05,   .062,  -0.04,   -0.04,
       0.,     0.,     0.,     0.,      0.,
       0.16,  -0.19,   0.003,  0.,      0.0001,
       0.,     0.,     0.,     0.,      0.,
       0.,     0.,     0.,     0.,      0.};
   t1 = std::chrono::high_resolution_clock::now();
   TMVA_SOFIE_rnn_sequence_batchwise::infer(input7);
   t2 = std::chrono::high_resolution_clock::now();
   duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
   std::cout << duration << "ms" << std::endl;
   float *y7 = TMVA_SOFIE_rnn_sequence_batchwise::tensor_Y;
   std::cout << "y" << std::endl;
   for (size_t i = 0; i < 9; i++) {
      for (size_t j = 0; j < 6; j++)
         std::cout << y7[i * 6 + j] << "  ";
      std::cout << std::endl;
   }
   // true y
   // -0.0160, -0.1818, -0.0401, -0.0794,  0.1761,  0.0137,
   // -0.4409, -0.5119, -0.1651,  0.0995,  0.8556, -0.4281,
   // -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
   // -0.1869,  0.8827, -0.6948, -0.2732,  0.4479,  0.9408,
   // -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
   // 0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
   // 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
   float* y_h7 = TMVA_SOFIE_rnn_sequence::tensor_Yh;
   std::cout << "y_h" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 6; j++)
         std::cout << y_h7[i * 6 + j] << "  ";
      std::cout << std::endl;
   }
   // true y_h
   // -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
   // -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
   // 0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097

}
